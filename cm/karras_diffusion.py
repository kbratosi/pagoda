"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from . import dist_util, logger
import torch.distributed as dist

from .nn import mean_flat, append_dims
import cm.script_util as script_util
from cm.enc_dec_lib import get_xl_feature, get_xl_logits



def get_weightings(weight_schedule, snrs, sigma_data, t, s, schedule_multiplier=None,):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=5.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    elif weight_schedule == "uniform_g":
        return 1./(1. - s / t) ** schedule_multiplier
    elif weight_schedule == "karras_weight":
        sigma = snrs ** -0.5
        weightings = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    elif weight_schedule == 'scaled_karras_weight':
        sigma = snrs ** -0.5
        weightings = (sigma ** 2 + (sigma_data ** 2) * ((1. - sigma) ** 2)) / (sigma * sigma_data) ** 2
    elif weight_schedule == "sq-t-inverse":
        weightings = 1. / snrs ** 0.25
    else:
        raise NotImplementedError()
    return weightings


class KarrasDenoiser:
    def __init__(
        self,
        args,
        schedule_sampler,
        diffusion_schedule_sampler,
        feature_extractor=None,
        discriminator_feature_extractor=None,
    ):
        self.args = args
        self.schedule_sampler = schedule_sampler
        self.diffusion_schedule_sampler = diffusion_schedule_sampler
        self.feature_extractor = feature_extractor
        self.discriminator_feature_extractor = discriminator_feature_extractor
        self.num_timesteps = args.start_scales
        self.dist = nn.MSELoss(reduction='none')
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max
        self.rho = args.rho
        self.sigma_min = args.sigma_min
        self.sigma_max = args.sigma_max
        self.sigma_data = args.sigma_data

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_c_in(self, sigma):
        alpha, beta = self.get_alpha_beta(sigma)
        c_in = 1 / (beta ** 2 + (alpha ** 2) * (self.sigma_data ** 2)) ** 0.5
        return c_in

    def get_ode_scalings(self, sigma):
        alpha, beta = self.get_alpha_beta(sigma)
        c_skip = alpha * (self.sigma_data ** 2) / (beta ** 2 + (alpha ** 2) * (self.sigma_data ** 2))
        c_out = beta * self.sigma_data / (beta ** 2 + (alpha ** 2) * (self.sigma_data ** 2)) ** 0.5
        return c_skip, c_out

    def get_decoder_scalings(self, sigma):
        alpha, beta = self.get_alpha_beta(sigma)
        alpha_min, beta_min = self.get_alpha_beta(self.sigma_min * th.ones_like(sigma))
        c_skip = alpha * (self.sigma_data ** 2) / ((beta - beta_min) ** 2 + (alpha ** 2) * (self.sigma_data ** 2))
        c_out = (beta - beta_min) * self.sigma_data / (beta ** 2 + (alpha ** 2) * (self.sigma_data ** 2)) ** 0.5
        return c_skip, c_out

    def beta_int(self, t):
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def get_alpha_beta(self, sigma):
        if self.args.diffusion_type == 'vp':
            alpha = th.exp(-0.5 * self.beta_int(sigma))
            beta = th.sqrt(1. - alpha ** 2)
        if self.args.diffusion_type == 'reflow':
            alpha = th.ones_like(sigma) - sigma
            beta = sigma
        if self.args.diffusion_type == 've':
            alpha = th.ones_like(sigma)
            beta = sigma
        return alpha, beta

    def get_diff_alpha_beta(self, sigma):
        if self.args.diffusion_type == 'vp':
            dlogalpha = -0.5 * self.beta(sigma)
            dlogbeta = 0.5 * self.beta(sigma) * th.exp(-self.beta_int(sigma)) / (1. - th.exp(-self.beta_int(sigma)))
        if self.args.diffusion_type == 'reflow':
            dlogalpha = -1. / (th.ones_like(sigma) - sigma)
            dlogbeta = 1. / sigma
        if self.args.diffusion_type == 've':
            dlogalpha = th.zeros_like(sigma)
            dlogbeta = 1. / sigma
        return dlogalpha, dlogbeta

    def get_encoder_c_in(self, sigma):
        return th.ones_like(sigma) / self.sigma_data

    def transform_from_edm_to_unit(self, t):
        return t / (t + 1.)

    def calculate_adaptive_weight(self, loss1, loss2, last_layer=None, loss1_grad_norm=0):
        if not loss1_grad_norm:
            loss1_grad_norm = th.norm(th.autograd.grad(loss1, last_layer, retain_graph=True)[0])
        loss2_grad = th.autograd.grad(loss2, last_layer, retain_graph=True)[0]
        #print("loss1_grad_norm: ", loss1_grad_norm)
        #print("loss2_grad_norm: ", th.norm(loss2_grad))
        d_weight = loss1_grad_norm / (th.norm(loss2_grad) + 1e-4)
        d_weight = th.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def rescaling_t(self, t):
        rescaled_t = 1000 * 0.25 * th.log(t + 1e-44)
        return rescaled_t

    def get_t(self, ind):
        if self.args.time_continuous:
            t = self.sigma_max ** (1 / self.rho) + ind * (
                    self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
            )
            t = t ** self.rho
        else:
            t = self.sigma_max ** (1 / self.rho) + ind / (self.args.start_scales - 1) * (
                    self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
            )
            t = t ** self.rho
        return t

    def get_num_heun_step(self, start_scales=-1, num_heun_step=-1, num_heun_step_random=None, heun_step_strategy='', time_continuous=None, start_heun_step=1,):
        if start_scales == -1:
            start_scales = self.args.start_scales
        if num_heun_step == -1:
            num_heun_step = self.args.hun_heun_step
        if num_heun_step_random == None:
            num_heun_step_random = self.args.num_heun_step_random
        if heun_step_strategy == '':
            heun_step_strategy = self.args.heun_step_strategy
        if time_continuous == None:
            time_continuous = self.args.time_continuous
        if num_heun_step_random:
            if time_continuous:
                num_heun_step = np.random.rand() * num_heun_step / start_scales
            else:
                if heun_step_strategy == 'uniform':
                    num_heun_step = np.random.randint(start_heun_step,1+num_heun_step)
                elif heun_step_strategy == 'weighted':
                    p = np.array([i ** self.args.heun_step_multiplier for i in range(1,1+num_heun_step)])
                    p = p / sum(p)
                    num_heun_step = np.random.choice([i+1 for i in range(len(p))], size=1, p=p)[0]
        else:
            if time_continuous:
                num_heun_step = num_heun_step / start_scales
            else:
                num_heun_step = num_heun_step
        return num_heun_step

    @th.no_grad()
    def heun_solver(self, ode, x, ind, dims, num_step=1, reverse=False, **model_kwargs):
        with th.no_grad():
            for k in range(num_step):
                if reverse:
                    t = self.get_t(ind - k)
                else:
                    t = self.get_t(ind + k)
                #if dist.get_rank() == 0:
                #    print("t, x: ", t[0].item(), (x ** 2).sum([1,2,3]).mean().item())

                denoiser = self.get_denoiser(ode, x, t, **model_kwargs)
                dlogalpha, dlogbeta = self.get_diff_alpha_beta(t)
                alpha_t, _ = self.get_alpha_beta(t)
                d = append_dims(dlogbeta, dims) * x + append_dims(dlogalpha - dlogbeta, dims) * append_dims(alpha_t, dims) * denoiser
                #d = (x - denoiser) / append_dims(t, dims)
                if reverse:
                    t2 = self.get_t(ind - k - 1)
                else:
                    t2 = self.get_t(ind + k + 1)
                x_phi_ODE_1st = x + d * append_dims(t2 - t, dims)
                denoiser2 = self.get_denoiser(ode, x_phi_ODE_1st, t2, **model_kwargs)
                dlogalpha, dlogbeta = self.get_diff_alpha_beta(t2)
                alpha_t, _ = self.get_alpha_beta(t2)
                next_d = append_dims(dlogbeta, dims) * x_phi_ODE_1st + append_dims(dlogalpha - dlogbeta, dims) * append_dims(alpha_t, dims) * denoiser2
                x_phi_ODE_2nd = x + (d + next_d) * append_dims((t2 - t) / 2, dims)
                x = x_phi_ODE_2nd
            return x

    def get_T_to_0(self, model, x_T, T=None, enable_grad=False, upsample_to_64=True, **model_kwargs):
        if enable_grad:
            grad = th.enable_grad()
        else:
            grad = th.no_grad()
        with grad:
            if T == None:
                T = th.ones(x_T.shape[0], device=x_T.device) * self.sigma_max
            #print("x_T, T: ", (x_T ** 2).mean().item(), (T ** 2).mean().item(), enable_grad, model_kwargs)
            c_in = append_dims(self.get_c_in(T), x_T.ndim)
            if self.args.decoder_style == 'stylegan':
                out = model((c_in * x_T).view(x_T.shape[0],-1), model_kwargs['y'])
                #assert not self.args.class_cond
            else:
                #print("1 x_T in get_T_to_0: ", x_T.shape)
                if self.args.upsample_to_64 and upsample_to_64:
                    x_T = F.interpolate(x_T, size=64, mode="bicubic")
                #print("2 x_T in get_T_to_0: ", x_T.shape)
                model_output = model(c_in * x_T, T, **model_kwargs)
                #c_skip, c_out = [append_dims(x, x_T.ndim) for x in self.get_decoder_scalings(T)]
                #print("c_skip: ", c_skip.shape, c_out.shape)
                out = model_output
                #out = c_out * model_output + c_skip * x_T
        return out

    def get_denoiser(self, ode, x_t, t, enable_grad=False, **model_kwargs):
        if enable_grad:
            grad = th.enable_grad()
        else:
            grad = th.no_grad()
        with grad:
            c_in = append_dims(self.get_c_in(t), x_t.ndim)
            model_output = ode(c_in * x_t, t, **model_kwargs)
            c_skip, c_out = [append_dims(x, x_t.ndim) for x in self.get_ode_scalings(t)]
            denoiser = c_out * model_output + c_skip * x_t
        return denoiser

    def check_isnan(self, loss):
        if th.isnan(loss.mean()):
            loss = th.zeros_like(loss)
            loss.requires_grad_(True)
        return loss

    def gather_all(self, sample, device='cpu'):
        gathered_sample = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_sample, sample)
        all_sample = [sample.cpu().numpy() for sample in gathered_sample]
        all_sample = th.from_numpy(np.concatenate(all_sample, axis=0)).to(device)
        return all_sample

    def get_last_layer(self, model, dataset='cifar10'):
        if dataset == 'cifar10':
            return model.module.model.dec[f'32x32_block{self.args.decoder_num_blocks}'].conv1.weight
        else:
            if self.args.decoder_style == 'unet':
                return model.module.output_blocks[self.last_layer_idx][0].out_layers[3].weight
            elif self.args.decoder_style == 'stylegan':
                return getattr(model.module.synthesis, self.last_layer_idx).affine.weight
            else:
                raise NotImplementedError

    def get_adaptive_weight(self, model, loss1, loss2, adaptive, loss1_grad_norm, type_):
        if adaptive:
            if loss1.sum():
                if self.args.data_name.lower() == 'cifar10':
                    if type_ == 'decoder':
                        if self.args.decoder_style == 'stylegan':
                            raise NotImplementedError
                        adaptive_weight = self.calculate_adaptive_weight(loss1.mean(),
                                                                  loss2.mean(),
                                                                  last_layer=self.get_last_layer(model, self.args.data_name))
                else:
                    if self.args.decoder_style == 'unet':
                        if loss1_grad_norm:
                            adaptive_weight = self.calculate_adaptive_weight(loss1.mean(),
                                                                             loss2.mean(),
                                                                             last_layer=self.get_last_layer(model, self.args.data_name),
                                                                             loss1_grad_norm=loss1_grad_norm)
                        else:
                            adaptive_weight = self.calculate_adaptive_weight(loss1.mean(),
                                                                      loss2.mean(),
                                                                      last_layer=self.get_last_layer(model, 'imagenet'))
                    else:
                        adaptive_weight = self.calculate_adaptive_weight(loss1.mean(),
                                                                         loss2.mean(),
                                                                         last_layer=model.module.get_last_layer())
                adaptive_weight = th.clip(adaptive_weight, 0.01, 10.)
            else:
                adaptive_weight = th.tensor(1., device=loss1.device)

        else:
            adaptive_weight = th.tensor(1., device=loss1.device)
        return adaptive_weight

    def null(self, x_start):
        loss = th.zeros(x_start.shape[0], device=x_start.device)
        loss.requires_grad_(True)
        return loss

    def get_decoder_discriminator_loss(self, decoder, discriminator, x_start, x_end, loss1, step, learn_generator=False, loss1_grad_norm=0,
                                       recon=False, **model_kwargs):
        if step < self.args.pretraining_step:
            return self.null(x_start)
        if learn_generator:
            if step % (2 * self.args.decoder_gan_frequency) == 0:
                if recon:
                    fake_x = self.fake_x
                else:
                    if self.args.pretrained_input_size == 32 and self.args.wavelet_input:
                        fake_noise = self.dwt_forward(th.randn((x_end.shape[0], x_end.shape[1], self.sampling_input_size, self.sampling_input_size), device=dist_util.dev()))[0] * self.sigma_max
                    else:
                        fake_noise = th.randn_like(x_end) * self.sigma_max
                    #print("fake_noise: ", fake_noise.shape)
                    fake_x = self.get_T_to_0(decoder, fake_noise, enable_grad=True, **model_kwargs)
                logits_fake = get_xl_feature(self.args, fake_x, feature_extractor=self.discriminator_feature_extractor,
                                                      discriminator=discriminator, step=step, decoder=True, **model_kwargs)
                g_loss = sum([(-l).mean() for l in logits_fake]) / len(logits_fake)
                adaptive_weight = self.get_adaptive_weight(decoder, loss1, g_loss, self.args.discriminator_adaptive_weight,
                                                           loss1_grad_norm=loss1_grad_norm, type_='decoder')
                if self.args.dev_log and step % self.args.log_interval == 0 and dist.get_rank() == 0:
                    logger.log(f"{'recon' if recon else 'decoder'} discriminator adaptive weight: ", adaptive_weight)
                discriminator_loss = adaptive_weight * g_loss
                if th.isnan(g_loss.mean()) or th.isnan(adaptive_weight):# or th.abs(th.sum(loss1)) < 1e-6:
                    discriminator_loss = self.null(x_start)
                if self.args.dev_log and step % self.args.log_interval == 0 and dist.get_rank() == 0:
                    logger.log(f"{'recon' if recon else 'decoder'} discriminator loss: ", discriminator_loss.mean().item(), g_loss.mean().item(), loss1.mean().item())
            else:
                return self.null(x_start)
        else:
            if recon:
                fake_x = self.fake_x.detach()
            else:
                if self.args.pretrained_input_size == 32 and self.args.wavelet_input:
                    fake_noise = self.dwt_forward(th.randn((x_start.shape[0], x_start.shape[1], self.sampling_input_size, self.sampling_input_size), device=dist_util.dev()))[0] * self.sigma_max
                else:
                    fake_noise = th.randn(self.x_end_shape, device=dist_util.dev()) * self.sigma_max
                fake_x = self.get_T_to_0(decoder, fake_noise, **model_kwargs)
            real_x = x_start
            logits_fake, logits_real = get_xl_feature(self.args, fake_x, target=real_x.detach(),
                                                      feature_extractor=self.discriminator_feature_extractor,
                                                      discriminator=discriminator, step=step, decoder=True, grad=th.no_grad, **model_kwargs)
            loss_Dgen = sum([(F.relu(th.ones_like(l) + l)).mean() for l in logits_fake]) / len(logits_fake)
            loss_Dreal = sum([(F.relu(th.ones_like(l) - l)).mean() for l in logits_real]) / len(logits_real)
            discriminator_loss = loss_Dreal + loss_Dgen
            discriminator_loss = self.check_isnan(discriminator_loss)
            if self.args.dev_log and step % self.args.log_interval == 1 and dist.get_rank() == 0:
                logger.log("decoder loss_Dgen: ", loss_Dgen)
                logger.log("decoder loss_Dreal: ", loss_Dreal)
            if self.args.save_png and step % self.args.save_period == -1:
                fake_x = self.gather_all(fake_x[:self.args.sampling_batch])
                real_x = self.gather_all(real_x[:self.args.sampling_batch])
                if dist.get_rank() == 0:
                    script_util.save(fake_x, logger.get_dir(), f'decoder_fake_x_{step}')  # _{r}')
                    script_util.save(real_x, logger.get_dir(), f'decoder_real_x_{step}')  # _{r}')
        return discriminator_loss

    def get_recon_loss(self, decoder, x_start, x_end, loss1, step, **model_kwargs):

        if step % (2 * self.args.decoder_gan_frequency) == 0:
            T_to_0_estimate = self.get_T_to_0(decoder, x_end, enable_grad=True, **model_kwargs)
            self.fake_x = T_to_0_estimate
            T_to_0_target = x_start
            #print("recon: ", T_to_0_target.shape, T_to_0_estimate.shape)
            #decoder_distill = ((T_to_0_target - T_to_0_estimate) ** 2).mean([1,2,3])
            if x_start.shape[2] < 256:
                T_to_0_estimate = F.interpolate(T_to_0_estimate, size=224, mode="bilinear")
                T_to_0_target = F.interpolate(T_to_0_target, size=224, mode="bilinear")
            if self.args.loss_norm == 'lpips':
                decoder_distill = (self.feature_extractor(
                    (T_to_0_estimate + 1) / 2.0,
                    (T_to_0_target + 1) / 2.0, ))
            elif self.args.loss_norm == 'cnn_vit':
                distances, estimate_features, target_features = [], [], []
                estimate_features, target_features = get_xl_feature(self.args, T_to_0_estimate, T_to_0_target,
                                                                    feature_extractor=self.feature_extractor, step=step)
                cnt = 0
                for _, _ in self.feature_extractor.items():
                    for fe in list(estimate_features[cnt].keys()):
                        norm_factor = th.sqrt(th.sum(estimate_features[cnt][fe] ** 2, dim=1, keepdim=True))
                        est_feat = estimate_features[cnt][fe] / (norm_factor + 1e-10)
                        norm_factor = th.sqrt(th.sum(target_features[cnt][fe] ** 2, dim=1, keepdim=True))
                        tar_feat = target_features[cnt][fe] / (norm_factor + 1e-10)
                        distances.append(self.dist(est_feat, tar_feat))
                    cnt += 1
                decoder_distill = th.cat([d.mean(dim=[2, 3]) for d in distances], dim=1).sum(dim=1)
            else:
                raise NotImplementedError
            decoder_distill = self.check_isnan(decoder_distill)
            if self.args.discriminator_adaptive_weight or self.args.decoder_adaptive_weight:
                if self.args.decoder_style == 'unet':
                    loss1_grad_norm = th.norm(th.autograd.grad(decoder_distill.mean(), self.get_last_layer(decoder, self.args.data_name), retain_graph=True)[0])
                else:
                    loss1_grad_norm = th.norm(th.autograd.grad(decoder_distill.mean(), decoder.module.get_last_layer(), retain_graph=True)[0])
            else:
                loss1_grad_norm = th.zeros_like(decoder_distill).mean()
            if self.args.save_png and step % self.args.save_period == 0:
                x_T = self.gather_all(x_end[:self.args.sampling_batch])
                T_to_0_estimate = self.gather_all(T_to_0_estimate[:self.args.sampling_batch])
                T_to_0_target = self.gather_all(T_to_0_target[:self.args.sampling_batch])
                if dist.get_rank() == 0:
                    script_util.save(x_T, logger.get_dir(), f'x_T_{step}')  # _{r}')
                    script_util.save(T_to_0_estimate, logger.get_dir(), f'T_to_0_{step}_estimate')  # _{r}')
                    script_util.save(T_to_0_target, logger.get_dir(), f'T_to_0_{step}_target')  # _{r}')
            return decoder_distill, loss1_grad_norm
        return self.null(x_start)

    def get_ode_loss(self, denoiser, x_start, step, **model_kwargs):
        t, _ = self.diffusion_schedule_sampler.sample(x_start.shape[0], dist_util.dev())
        #print("t: ", t[:100])
        dims = x_start.ndim
        alpha_t, beta_t = self.get_alpha_beta(t)
        #print("alpha_t: ", alpha_t[:100])
        #print("beta_t: ", beta_t[:100])
        x_t = append_dims(alpha_t, dims) * x_start + append_dims(beta_t, dims) * th.randn_like(x_start)
        denoised = self.get_denoiser(denoiser, x_t, t, enable_grad=True, **model_kwargs)
        snrs = (alpha_t / beta_t) ** 2
        denoising_weights = append_dims(
            get_weightings(self.args.diffusion_weight_schedule, snrs, self.sigma_data, None, None), dims)
        ode_loss = mean_flat(denoising_weights * (denoised - x_start) ** 2)
        ode_loss = self.check_isnan(ode_loss)
        if self.args.save_png and step % self.args.save_period == 0:
            x_t = self.gather_all(x_t[:self.args.sampling_batch])
            denoised = self.gather_all(denoised[:self.args.sampling_batch])
            x_start = self.gather_all(x_start[:self.args.sampling_batch])
            if dist.get_rank() == 0:
                script_util.save(x_t, logger.get_dir(), f'x_t_{step}')  # _{r}')
                script_util.save(denoised, logger.get_dir(), f'denoiser_{step}')  # _{r}')
                script_util.save(x_start, logger.get_dir(), f'x_start_{step}')  # _{r}')
        return ode_loss

    def get_enc_loss(self, encoder, decoder, x_start, x_end, x_start_eval, x_end_eval, model_kwargs_eval, step, **model_kwargs):
        noise_estimate = self.get_T_to_0(encoder, x_start, T=th.zeros(x_end.shape[0], device=x_end.device), enable_grad=True, **model_kwargs)
        noise_target = x_end / self.args.sigma_max
        encoder_distill = ((noise_estimate - noise_target) ** 2).mean() + (noise_estimate - noise_target).abs().mean()
        decoder_distill = self.check_isnan(encoder_distill)
        if self.args.save_png and step % self.args.save_period == 0:
            original_train = self.gather_all(x_start[:self.args.sampling_batch])
            reconstruct_train = self.gather_all(self.get_T_to_0(decoder, noise_estimate * self.args.sigma_max, **model_kwargs)[:self.args.sampling_batch])
            original_eval = self.gather_all(x_start_eval[:self.args.sampling_batch])
            reconstruct_eval = self.gather_all(
                self.get_T_to_0(decoder, self.get_T_to_0(encoder, x_start_eval, T=th.zeros(x_end_eval.shape[0], device=x_end_eval.device), enable_grad=False, **model_kwargs_eval) * self.args.sigma_max, **model_kwargs_eval)[:self.args.sampling_batch])
            noise_estimate = self.gather_all(noise_estimate[:self.args.sampling_batch])
            noise_target = self.gather_all(noise_target[:self.args.sampling_batch])
            if dist.get_rank() == 0:
                script_util.save(original_train, logger.get_dir(), f'train_data_{step}_original')
                script_util.save(reconstruct_train, logger.get_dir(), f'train_data_{step}_reconstruct')
                script_util.save(original_eval, logger.get_dir(), f'eval_data_{step}_original')
                script_util.save(reconstruct_eval, logger.get_dir(), f'eval_data_{step}_reconstruct')
                script_util.save(noise_estimate, logger.get_dir(), f'0_to_T_noise_{step}_estimate')
                script_util.save(noise_target, logger.get_dir(), f'0_to_T_noise_{step}_target')
        return decoder_distill

    def pgd_losses(
        self,
        step,
        encoder,
        decoder,
        x_start,
        model_kwargs=None,
        recon_discriminator=None,
        decoder_discriminator=None,
        init_step=0,
        loss1_grad_norm=0,
        mode='reconstruction',
    ):

        #print(x_start.shape)
        #import sys
        #sys.exit()
        if model_kwargs is None:
            model_kwargs = {}
        terms = {}
        x_end = model_kwargs.pop('z').permute(0, 3, 1, 2).contiguous()
        self.x_end_shape = x_end.shape
        assert x_start.dtype == x_end.dtype
        dropout_state = th.get_rng_state()
        th.set_rng_state(dropout_state)
        #print("x_start: ", x_start.shape)
        #print("x_end: ", x_end.shape)
        if step % 2 == 0:
            if self.args.separate_update:
                if mode == 'reconstruction':
                    if encoder != None:
                        with th.no_grad():
                            x_end = self.get_T_to_0(encoder, F.interpolate(x_start, size=32, mode="bicubic", antialias=True, align_corners=False), T=th.zeros(x_end.shape[0], device=x_end.device),
                                                    enable_grad=False, upsample_to_64=False, **model_kwargs) * self.args.sigma_max
                    terms['decoder_recon_loss'], terms['loss1_grad_norm'] = self.get_recon_loss(decoder, x_start, x_end, self.null(x_start), step, **model_kwargs)
                    if self.args.recon_discriminator:
                        terms['recon_discriminator_loss'] = self.get_decoder_discriminator_loss(decoder,
                                recon_discriminator, x_start, x_end, terms['decoder_recon_loss'], step - init_step,
                                learn_generator=True, recon=True, loss1_grad_norm=terms['loss1_grad_norm'], **model_kwargs)
                elif mode == 'gan':
                    terms['discriminator_loss'] = self.get_decoder_discriminator_loss(decoder, decoder_discriminator,
                    x_start, x_end, th.ones_like(self.null(x_start)) if self.args.decoder_adaptive_weight else self.null(x_start),
                    step - init_step, learn_generator=True, loss1_grad_norm=loss1_grad_norm, recon=False, **model_kwargs)
                elif mode == 'discriminator':
                    terms['recon_discriminator_loss'] = self.get_decoder_discriminator_loss(decoder, recon_discriminator,
                        x_start, x_end, None, step - init_step, learn_generator=False, recon=True, **model_kwargs)
            else:
                if mode == 'reconstruction':
                    if encoder != None:
                        with th.no_grad():
                            x_end = self.get_T_to_0(encoder, F.interpolate(x_start, size=32, mode="bicubic", antialias=True, align_corners=False), T=th.zeros(x_end.shape[0], device=x_end.device),
                                                    enable_grad=False, upsample_to_64=False, **model_kwargs) * self.args.sigma_max
                    terms['decoder_recon_loss'], terms['loss1_grad_norm'] = self.get_recon_loss(decoder, x_start, x_end, self.null(x_start), step, **model_kwargs)
                    terms['discriminator_loss'] = self.get_decoder_discriminator_loss(decoder, decoder_discriminator,
                           x_start, x_end, terms['decoder_recon_loss'], step - init_step, learn_generator=True, recon=False,
                            loss1_grad_norm=terms['loss1_grad_norm'], **model_kwargs)
                    if self.args.recon_discriminator:
                        terms['recon_discriminator_loss'] = self.get_decoder_discriminator_loss(decoder, recon_discriminator,
                                    x_start, x_end, terms['decoder_recon_loss'], step - init_step, learn_generator=True, recon=True,
                            loss1_grad_norm=terms['loss1_grad_norm'], **model_kwargs)
                elif mode == 'discriminator':
                    terms['recon_discriminator_loss'] = self.get_decoder_discriminator_loss(decoder, recon_discriminator,
                        x_start, x_end, None, step - init_step, learn_generator=False, recon=True, **model_kwargs)
                else:
                    raise NotImplementedError
        if step % 2 == 1:
            terms['discriminator_loss'] = self.get_decoder_discriminator_loss(decoder, decoder_discriminator,
                                       x_start, None, None, step - init_step, learn_generator=False, recon=False, **model_kwargs)

        return terms

    def diffusion_losses(
        self,
        step,
        denoiser,
        x_start,
        model_kwargs=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        terms = {}
        dropout_state = th.get_rng_state()
        th.set_rng_state(dropout_state)
        terms['diffusion_loss'] = self.get_ode_loss(denoiser, x_start, step, **model_kwargs)
        return terms

    def encoder_losses(
        self,
        step,
        encoder,
        decoder,
        x_start,
        model_kwargs=None,
        x_start_eval=None,
        model_kwargs_eval=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        terms = {}
        x_end = model_kwargs.pop('z').permute(0, 3, 1, 2).contiguous()
        if model_kwargs_eval != None:
            x_end_eval = model_kwargs_eval.pop('z').permute(0, 3, 1, 2).contiguous()
        self.x_end_shape = x_end.shape
        assert x_start.dtype == x_end.dtype
        dropout_state = th.get_rng_state()
        th.set_rng_state(dropout_state)
        terms['encoder_loss'] = self.get_enc_loss(encoder, decoder, x_start, x_end, x_start_eval, x_end_eval, model_kwargs_eval, step, **model_kwargs)
        return terms
