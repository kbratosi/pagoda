import copy
import functools
import os
import cv2
from skimage.metrics import structural_similarity as SSIM_
import shutil
import gc

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
import torch.nn.functional as F

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
import nvidia_smi

from .fp16_util import (
    get_param_groups_and_shapes,
    get_target_param_groups_and_shapes,
    make_master_params,
    state_dict_to_master_params,
    master_params_to_model_params,
)
import numpy as np
from cm.sample_util import karras_sample
from cm.random_util import get_generator
from torchvision.utils import make_grid, save_image
import datetime
import dnnlib
import pickle
import glob
import scipy
import time

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        decoder,
        ode,
        recon_discriminator,
        decoder_discriminator,
        diffusion,
        data,
        data_for_GAN,
        batch_size,
        args=None,
    ):
        self.args = args
        #if self.args.gpu_usage:
        nvidia_smi.nvmlInit()
        self.deviceCount = nvidia_smi.nvmlDeviceGetCount()
        self.print_gpu_usage('Before everything')
        self.decoder = decoder
        self.ode = ode
        if self.args.sanity_check:
            for name, param in self.decoder.named_parameters():
                logger.log("check and understand how decoder override decoder parameters")
                logger.log("decoder parameter before overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                break
        self.recon_discriminator = recon_discriminator
        self.decoder_discriminator = decoder_discriminator
        self.diffusion = diffusion
        self.data = data
        self.data_for_GAN = data_for_GAN
        self.batch_size = batch_size
        self.microbatch = args.microbatch if args.microbatch > 0 else batch_size
        self.step = 0
        self.resume_step = 0
        self.loss1_grad_norm = 0
        self.ema_rate = (
            [args.ema_rate]
            if isinstance(args.ema_rate, float)
            else [float(x) for x in args.ema_rate.split(",")]
        )
        self.global_batch = self.batch_size * dist.get_world_size()
        self.fids = []
        self.generator = get_generator('determ', self.args.eval_num_samples, self.args.eval_seed, dwt=(self.args.pretrained_input_size == 32 and self.args.wavelet_input))
        if self.args.pretrained_input_size == 32 and self.args.wavelet_input:
            self.sampling_input_size = self.args.input_size * 2
            from pytorch_wavelets import DWTForward
            self.diffusion.dwt_forward = DWTForward(J=1, mode='zero', wave='haar').to(dist_util.dev())
        else:
            self.sampling_input_size = self.args.input_size
        self.diffusion.sampling_input_size = self.sampling_input_size
        self.x_T = self.generator.randn(*(self.args.sampling_batch, self.args.in_channels, self.sampling_input_size, self.sampling_input_size),
                                        device=dist_util.dev()) * self.args.sigma_max #.to(dist_util.dev())
        self.training_mode = self.args.training_mode
        self.total_training_steps = self.args.total_training_steps

        self.diffusion.ode = self.ode
        self.global_step = self.step
        self.initial_step = copy.deepcopy(self.step)
        print("self.x_T: ", (self.x_T ** 2).sum([1, 2, 3]).mean().item())
        if self.args.class_cond:
            self.classes = self.generator.randint(0, self.args.num_classes, (self.args.sampling_batch,), device=dist_util.dev())
            if self.args.data_name.lower() == 'cifar10':
                self.classes.sort()

        self.sync_cuda = th.cuda.is_available()

        self.print_gpu_usage('Before model loading')
        self.mp_decoder_trainer, self.opt_dec, self.ddp_decoder, = \
            self.load_model(self.decoder, 'decoder', self.args.lr_dec, self.args.use_fp16, self.ema_rate)
        self.print_gpu_usage('After decoder loading')
        self.mp_decoder_discriminator_trainer, self.opt_decoder_disc, self.ddp_decoder_discriminator, = \
            self.load_model(self.decoder_discriminator, 'decoder_discriminator', self.args.lr_disc, False, None)
        self.print_gpu_usage('After discriminator loading')
        self.mp_recon_discriminator_trainer, self.opt_recon_disc, self.ddp_recon_discriminator, = \
            self.load_model(self.recon_discriminator, 'recon_discriminator', self.args.lr_disc, False, None)
        self.print_gpu_usage('After discriminator2 loading')
        self.step = self.resume_step

    def load_model(self, model, type_, lr, use_fp16=True, ema_rate=None):
        mp_trainer, optimizer, ddp_model = None, None, None
        if model != None:
            self._load_and_sync_parameters(model, type_)
            mp_trainer = MixedPrecisionTrainer(
                model=model, use_fp16=use_fp16, fp16_scale_growth=self.args.fp16_scale_growth, )
            optimizer = RAdam(mp_trainer.master_params, lr=lr,
                                 weight_decay=self.args.weight_decay)
            self._load_optimizer_state(optimizer, type_)
            #ema_params = [self._load_ema_parameters(mp_trainer, rate, type_) for rate in ema_rate] if ema_rate != [] else None
            if th.cuda.is_available():
                self.use_ddp = True
                ddp_model = DDP(
                    model,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
            else:
                raise NotImplementedError
            if ema_rate != None:
                self.ema_params = [
                    self._load_ema_parameters(mp_trainer, rate) for rate in ema_rate
                ]
        return mp_trainer, optimizer, ddp_model

    def _load_and_sync_parameters(self, model, type_):
        if model != None:
            resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
            path, name = os.path.split(resume_checkpoint)
            target_name = name.replace("decoder", type_)
            resume_checkpoint = os.path.join(path, target_name)

            if os.path.exists(resume_checkpoint):
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint, type_)

                if dist.get_rank() == 0:
                    logger.log(f"loading pretrained {type_} from checkpoint: {resume_checkpoint}...")
                    if self.args.map_location == 'cuda':
                        state_dict = th.load(resume_checkpoint, map_location=dist_util.dev())
                    else:
                        state_dict = dist_util.load_state_dict(resume_checkpoint, map_location='cpu')
                    model.load_state_dict(state_dict, strict=True)
                    logger.log(f"end loading pretrained {type_} from checkpoint: {resume_checkpoint}...")
                dist_util.sync_params(model.parameters())
                dist_util.sync_params(model.buffers())
            logger.log(f"end synchronizing pretrained {type_} from GPU0 to all GPUs")
        else:
            logger.log(f"No {type_} loaded")

    def _load_optimizer_state(self, opt, type_='encoder'):
        if opt != None and self.resume_step:
            main_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
            opt_checkpoint = bf.join(
                bf.dirname(main_checkpoint), f"{type_}_opt_{self.resume_step:06}.pt"
            )
            if bf.exists(opt_checkpoint):
                logger.log(f"loading {type_} optimizer state from checkpoint: {opt_checkpoint}")
                if self.args.map_location == 'cuda':
                    state_dict = dist_util.load_state_dict(
                        opt_checkpoint, map_location=dist_util.dev()
                    )
                else:
                    state_dict = th.load(opt_checkpoint, map_location="cpu")
                opt.load_state_dict(state_dict)
                logger.log(f"end loading {type_} optimizer state from checkpoint: {opt_checkpoint}")
            else:
                logger.log(f"{type_} optimizer state does not exist at: {opt_checkpoint}")

    def _load_ema_parameters(self, mp_trainer, rate):
        ema_params = copy.deepcopy(mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, 'decoder', rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                if self.args.map_location == 'cuda':
                    state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())#"cpu")
                else:
                    state_dict = dist_util.load_state_dict(
                        ema_checkpoint, map_location='cpu'
                    )
                ema_params = mp_trainer.state_dict_to_master_params(state_dict)
                logger.log(f"end loading EMA from checkpoint: {ema_checkpoint}...")

            dist_util.sync_params(ema_params)
        logger.log(f"end synchronizing EMA from GPU0 to all GPUs")

        return ema_params

    def print_gpu_usage(self, prefix=''):
        for i in range(self.deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            logger.log(
                f"{prefix} |Device {i}| Mem Free: {mem.free / 1024 ** 2:5.2f}MB / {mem.total / 1024 ** 2:5.2f}MB | gpu-util: {util.gpu / 100.0:3.1%} | gpu-mem: {util.memory / 100.0:3.1%} |")


    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_decoder_trainer.master_params, rate=rate)

    def update_parameter(self, mp_trainer, opt, ema_rate):
        took_step = mp_trainer.optimize(opt)
        if took_step:
            self._update_ema()

    def sampling(self, model, sampler, ctm=None, teacher=False, step=-1, num_samples=-1, batch_size=-1, rate=0.999,
                 png=False, resize=True, generator=None, class_generator=None, sample_dir='', enc_x_T=None, fid_eval=False, random_class=True):
        if not teacher:
            model.eval()
        if step == -1:
            step = self.args.sampling_steps
        if batch_size == -1:
            batch_size = self.args.sampling_batch

        number = 0
        while num_samples > number:

            with th.no_grad():
                model_kwargs = {}
                if self.args.class_cond:
                    if self.args.train_classes >= 0:
                        classes = th.ones(size=(batch_size,), device=dist_util.dev(), dtype=int) * self.args.train_classes
                        model_kwargs["y"] = classes
                    elif self.args.train_classes == -2:
                        classes = [0, 1, 9, 11, 29, 31, 33, 55, 76, 89, 90, 130, 207, 250, 279, 281, 291, 323, 386, 387,
                                   388, 417, 562, 614, 759, 789, 800, 812, 848, 933, 973, 980]
                        assert batch_size % len(classes) == 0
                        model_kwargs["y"] = th.tensor([x for x in classes for _ in range(batch_size // len(classes))], device=dist_util.dev())
                    else:
                        if class_generator != None:
                            model_kwargs["y"] = class_generator.randint(0, self.args.num_classes, (batch_size,), device=dist_util.dev())
                        else:
                            if num_samples == -1:
                                model_kwargs["y"] = self.classes.to(dist_util.dev())
                            else:
                                model_kwargs["y"] = th.randint(0, self.args.num_classes, size=(batch_size, ), device=dist_util.dev())
                        if not random_class:
                            model_kwargs["y"] = self.classes.to(dist_util.dev())
                    print("sampling classes: ", model_kwargs["y"])
                if generator != None:
                    x_T = generator.randn(*(batch_size, self.args.in_channels, self.sampling_input_size, self.sampling_input_size),
                                device=dist_util.dev()) * self.args.sigma_max
                    if self.args.large_log:
                        print("x_T: ", x_T[0][0][0][:3])
                else:
                    x_T = None

                #model_kwargs["y"] = th.zeros_like(model_kwargs["y"])
                #print("self.x_T: ", (self.x_T ** 2).sum([1,2,3]).mean().item())
                sample = karras_sample(
                    diffusion=self.diffusion,
                    model=model,
                    shape=(batch_size, self.args.in_channels, self.sampling_input_size, self.sampling_input_size),
                    steps=step,
                    model_kwargs=model_kwargs,
                    device=dist_util.dev(),
                    clip_denoised=True if teacher else self.args.clip_denoised,
                    sampler=sampler,
                    generator=None,
                    teacher=teacher,
                    ctm=ctm if ctm != None else True if self.args.training_mode.lower() == 'ctm' else False,
                    x_T=None if fid_eval else x_T if generator != None else enc_x_T if enc_x_T != None else self.x_T.to(dist_util.dev()), # if num_samples == -1 else None,
                    clip_output=self.args.clip_output,
                    sigma_min=self.args.sigma_min,
                    sigma_max=self.args.sigma_max,
                    rho=self.args.rho,
                    train=False,
                    dwt = (self.args.pretrained_input_size == 32 and self.args.wavelet_input),
                )
                if resize:
                    sample = F.interpolate(sample, size=224, mode="bilinear")

                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)
                all_images = [sample.cpu().numpy() for sample in gathered_samples]
                arr = np.concatenate(all_images, axis=0)
                if dist.get_rank() == 0:
                    os.makedirs(bf.join(get_blob_logdir(), f"{sample_dir}"), exist_ok=True)
                    if self.args.large_log:
                        print(f"saving to {bf.join(get_blob_logdir(), sample_dir)}")
                    nrow = int(np.sqrt(arr.shape[0]))
                    image_grid = make_grid(th.tensor(arr).permute(0, 3, 1, 2) / 255., nrow, padding=2)
                    if num_samples == -1:
                        with bf.BlobFile(bf.join(get_blob_logdir(), f"{'teacher_' if teacher else ''}sample_{sampler}_sampling_step_{step}_step_{self.step}.png"), "wb") as fout:
                            save_image(image_grid, fout)
                    else:
                        if generator != None:
                            os.makedirs(bf.join(get_blob_logdir(), sample_dir),
                                        exist_ok=True)
                            if png or number <= 3000:
                                with bf.BlobFile(bf.join(get_blob_logdir(), sample_dir,
                                                         f"sample_{number // arr.shape[0]}.png"), "wb") as fout:
                                    save_image(image_grid, fout)
                            if not png:
                                np.savez(bf.join(get_blob_logdir(), sample_dir, f"sample_{number // arr.shape[0]}.npz"),
                                         arr)

                        else:
                            r = np.random.randint(100000000)
                            if self.args.large_log:
                                logger.log(f'{dist.get_rank()} number {number}')
                            os.makedirs(bf.join(get_blob_logdir(), f"{sample_dir}"),
                                        exist_ok=True)
                            if png or number <= 1000:
                                with bf.BlobFile(bf.join(get_blob_logdir(), sample_dir,
                                                         f"sample_{r}.png"), "wb") as fout:
                                    save_image(image_grid, fout)
                            if not png:
                                np.savez(bf.join(get_blob_logdir(), sample_dir, f"sample_{r}.npz"),
                                         arr)


                number += arr.shape[0]
                print(f"{number} number samples complete")
        if not teacher:
            model.train()

    def calculate_similarity_metrics(self, image_path, num_samples=50000, step=1, batch_size=100, rate=0.999, sampler='exact', log=True):
        files = glob.glob(os.path.join(image_path, 'sample*.npz'))
        files.sort()
        count = 0
        psnr = 0
        ssim = 0
        for i, file in enumerate(files):
            images = np.load(file)['arr_0']
            for k in range((images.shape[0] - 1) // batch_size + 1):
                #ref_img = self.ref_images[count + k * batch_size: count + (k + 1) * batch_size]
                if count + batch_size > num_samples:
                    remaining_num_samples = num_samples - count
                else:
                    remaining_num_samples = batch_size
                img = images[k * batch_size: k * batch_size + remaining_num_samples]
                ref_img = self.ref_images[count: count + remaining_num_samples]
                psnr += cv2.PSNR(img, ref_img) * remaining_num_samples
                ssim += SSIM_(img,ref_img,multichannel=True,channel_axis=3,data_range=255) * remaining_num_samples
                count = count + remaining_num_samples
                print(count)
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
        assert count == num_samples
        print(count)
        psnr /= num_samples
        ssim /= num_samples
        assert num_samples % 1000 == 0
        if log:
            logger.log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate} PSNR-{num_samples // 1000}k: {psnr}, SSIM-{num_samples // 1000}k: {ssim}")
        else:
            return psnr, ssim

    def calculate_inception_stats(self, data_name, image_path, num_samples=50000, batch_size=100, device=th.device('cuda')):
        if data_name.lower() == 'cifar10':
            print(f'Loading images from "{image_path}"...')
            mu = th.zeros([self.feature_dim], dtype=th.float64, device=device)
            sigma = th.zeros([self.feature_dim, self.feature_dim], dtype=th.float64, device=device)
            files = glob.glob(os.path.join(image_path, 'sample*.npz'))
            count = 0
            for file in files:
                images = np.load(file)['arr_0']  # [0]#["samples"]
                for k in range((images.shape[0] - 1) // batch_size + 1):
                    mic_img = images[k * batch_size: (k + 1) * batch_size]
                    mic_img = th.tensor(mic_img).permute(0, 3, 1, 2).to(device)
                    features = self.detector_net(mic_img, **self.detector_kwargs).to(th.float64)
                    if count + mic_img.shape[0] > num_samples:
                        remaining_num_samples = num_samples - count
                    else:
                        remaining_num_samples = mic_img.shape[0]
                    mu += features[:remaining_num_samples].sum(0)
                    sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
                    count = count + remaining_num_samples
                    print(count)
                    if count >= num_samples:
                        break
                if count >= num_samples:
                    break
            assert count == num_samples
            print(count)
            mu /= num_samples
            sigma -= mu.ger(mu) * num_samples
            sigma /= num_samples - 1
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()
            return mu, sigma
        else:
            filenames = glob.glob(os.path.join(image_path, '*.npz'))
            imgs = []
            for file in filenames:
                try:
                    img = np.load(file)  # ['arr_0']
                    try:
                        img = img['data']
                    except:
                        img = img['arr_0']
                    imgs.append(img)
                except:
                    pass
            imgs = np.concatenate(imgs, axis=0)
            os.makedirs(os.path.join(image_path, 'single_npz'), exist_ok=True)
            np.savez(os.path.join(os.path.join(image_path, 'single_npz'), f'data'),
                     imgs)  # , labels)
            logger.log("computing sample batch activations...")
            sample_acts = self.evaluator.read_activations(
                os.path.join(os.path.join(image_path, 'single_npz'), f'data.npz'))
            logger.log("computing/reading sample batch statistics...")
            sample_stats, sample_stats_spatial = tuple(self.evaluator.compute_statistics(x) for x in sample_acts)
            with open(os.path.join(os.path.join(image_path, 'single_npz'), f'stats'), 'wb') as f:
                pickle.dump({'stats': sample_stats, 'stats_spatial': sample_stats_spatial}, f)
            with open(os.path.join(os.path.join(image_path, 'single_npz'), f'acts'), 'wb') as f:
                pickle.dump({'acts': sample_acts[0], 'acts_spatial': sample_acts[1]}, f)
            return sample_acts, sample_stats, sample_stats_spatial

    def compute_fid(self, mu, sigma, ref_mu=None, ref_sigma=None):
        if np.array(ref_mu == None).sum():
            ref_mu = self.mu_ref
            assert ref_sigma == None
            ref_sigma = self.sigma_ref
        print("mu: ", mu.mean())
        print("ref_mu: ", ref_mu.mean())
        print("sigma: ", (sigma ** 2).mean())
        print("ref_sigma: ", (ref_sigma ** 2).mean())
        m = np.square(mu - ref_mu).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, ref_sigma), disp=False)
        fid = m + np.trace(sigma + ref_sigma - s * 2)
        fid = float(np.real(fid))
        return fid

    def calculate_inception_stats_npz(self, image_path, num_samples=50000, step=1, batch_size=100, device=th.device('cuda'),
                                      rate=0.999):
        print(f'Loading images from "{image_path}"...')
        mu = th.zeros([self.feature_dim], dtype=th.float64, device=device)
        sigma = th.zeros([self.feature_dim, self.feature_dim], dtype=th.float64, device=device)

        files = glob.glob(os.path.join(image_path, 'sample*.npz'))
        count = 0
        for file in files:
            images = np.load(file)['arr_0']  # [0]#["samples"]
            for k in range((images.shape[0] - 1) // batch_size + 1):
                mic_img = images[k * batch_size: (k + 1) * batch_size]
                mic_img = th.tensor(mic_img).permute(0, 3, 1, 2).to(device)
                features = self.detector_net(mic_img, **self.detector_kwargs).to(th.float64)
                if count + mic_img.shape[0] > num_samples:
                    remaining_num_samples = num_samples - count
                else:
                    remaining_num_samples = mic_img.shape[0]
                mu += features[:remaining_num_samples].sum(0)
                sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
                count = count + remaining_num_samples
                logger.log(count)
            if count >= num_samples:
                break
        assert count == num_samples
        print(count)
        mu /= num_samples
        sigma -= mu.ger(mu) * num_samples
        sigma /= num_samples - 1
        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()

        m = np.square(mu - self.mu_ref).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, self.sigma_ref), disp=False)
        fid = m + np.trace(sigma + self.sigma_ref - s * 2)
        fid = float(np.real(fid))
        assert num_samples % 1000 == 0
        logger.log(f"{self.step}-th step exact sampler (NFE {step}) EMA {rate} FID-{num_samples // 1000}k: {fid}")

    def save(self, save_full=True, only_model=False):
        def save_checkpoint(rate, params, trainer, type_):
            state_dict = trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving decoder {rate}...")
                if not rate:
                    filename = f"{type_}_{self.global_step:06d}.pt"
                else:
                    filename = f"{type_}_ema_{rate}_{self.global_step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        if only_model:
            for rate, params in zip(self.ema_rate, self.ema_params):
                if rate == 0.999:
                    save_checkpoint(0, params, self.mp_decoder_trainer, 'decoder')
        else:
            for rate, params in zip(self.ema_rate, self.ema_params):
                if not save_full:
                    if rate == 0.999:
                        save_checkpoint(rate, params, self.mp_decoder_trainer, 'decoder')
                else:
                    save_checkpoint(rate, params, self.mp_decoder_trainer, 'decoder')

            logger.log("saving optimizer state...")
            if dist.get_rank() == 0:
                with bf.BlobFile(bf.join(get_blob_logdir(), f"decoder_opt_{self.global_step:06d}.pt"), "wb",) as f:
                    th.save(self.opt_dec.state_dict(), f)
                if self.args.decoder_discriminator_training:
                    with bf.BlobFile(bf.join(get_blob_logdir(), f"decoder_discriminator_opt_{self.global_step:06d}.pt"), "wb",) as f:
                        th.save(self.opt_decoder_disc.state_dict(), f)
                    if not os.path.exists(os.path.join(get_blob_logdir(), 'feature_extractor')):
                        with bf.BlobFile(bf.join(get_blob_logdir(), f"feature_extractor.pt"), "wb", ) as f:
                            th.save(self.diffusion.discriminator_feature_extractor.state_dict(), f)

            # Save model parameters last to prevent race conditions where a restart
            # loads model at step N, but opt/ema state isn't saved for step N.
            save_checkpoint(0, self.mp_decoder_trainer.master_params, self.mp_decoder_trainer, 'decoder')
            if self.args.decoder_discriminator_training:
                save_checkpoint(0, self.mp_decoder_discriminator_trainer.master_params, self.mp_decoder_discriminator_trainer, 'decoder_discriminator')
        dist.barrier()

class CMTrainLoop(TrainLoop):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if self.args.data_name.lower() == 'cifar10':
            if dist.get_rank() == 0:
                print('Loading Inception-v3 model...')
                detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
                self.detector_kwargs = dict(return_features=True)
                self.feature_dim = 2048
                with dnnlib.util.open_url(detector_url, verbose=(0 == 0)) as f:
                    self.detector_net = pickle.load(f).to(dist_util.dev())
                with dnnlib.util.open_url(self.args.ref_path) as f:
                    ref = dict(np.load(f))
                self.mu_ref = ref['mu']
                self.sigma_ref = ref['sigma']
                if self.args.check_dm_performance:
                    if self.args.ae_image_path_seed_42 != '':
                        self.ae_mu, self.ae_sigma = self.calculate_inception_stats(self.args.data_name,
                                                                                   self.args.ae_image_path_seed_42,
                                                                                   num_samples=self.args.eval_num_samples)
                    self.dm_mu, self.dm_sigma = self.calculate_inception_stats(self.args.data_name,
                                                                               self.args.dm_sample_path_seed_42,
                                                                               num_samples=self.args.eval_num_samples)
                    logger.log(f"DM FID-50k: {self.compute_fid(self.dm_mu, self.dm_sigma)}")
                    ref_files = glob.glob(os.path.join(self.args.dm_sample_path_seed_42, 'sample*.npz'))
                    ref_files.sort()
                    self.ref_images = []
                    for i, ref_file in enumerate(ref_files):
                        ref_images = np.load(ref_file)['arr_0']
                        self.ref_images.append(ref_images)
                    self.ref_images = np.concatenate(self.ref_images)
                    if self.args.ae_image_path_seed_42 != '':
                        logger.log(f"Regenerated DM Samples (by LSGM AE) FID-50k: {self.compute_fid(self.ae_mu, self.ae_sigma)}")
                        psnr, ssim = self.calculate_similarity_metrics(
                            self.args.ae_image_path_seed_42, num_samples=self.args.eval_num_samples, step=1,
                            rate=0.0, sampler='LSGM Auto-Encoder', log=False)
                        logger.log(f"Regenerated DM Samples (by LSGM AE) PSNR-50k: {psnr}, SSIM-10k: {ssim}")
        else:
            if self.args.decoder_style == 'unet':
                for name, params in self.decoder.named_parameters():
                    if name.split('.')[0] == 'output_blocks':
                        last_layer_idx = int(name.split('.')[1])
            elif self.args.decoder_style == 'stylegan':
                for name, params in self.decoder.named_parameters():
                    if name.split('.')[0] == 'synthesis' and name.split('.')[1] != 'input':
                        if int(name.split('.')[1].split('_')[2]) != 3:
                            last_layer_idx = name.split('.')[1]
            elif self.args.decoder_style == 'ldm':
                last_layer_idx = -1
            else:
                raise NotImplementedError
            self.diffusion.last_layer_idx = last_layer_idx
            logger.log("decoder last layer index: ", last_layer_idx)
            if self.args.eval_fid:
                global tf
                global Evaluator
                import tensorflow.compat.v1 as tf
                from cm.evaluator import Evaluator
            if dist.get_rank() == 0:
                # if self.args.eval_fid or self.args.eval_similarity:
                #     #import tensorflow.compat.v1 as tf
                #     #from cm.evaluator import Evaluator
                #     config = tf.ConfigProto(
                #         allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
                #     )
                #     config.gpu_options.allow_growth = True
                #     config.gpu_options.per_process_gpu_memory_fraction = 0.1
                #     self.evaluator = Evaluator(tf.Session(config=config), batch_size=100)
                #     self.ref_acts = self.evaluator.read_activations(self.args.ref_path)
                #     self.ref_stats, self.ref_stats_spatial = self.evaluator.read_statistics(self.args.ref_path, self.ref_acts)
                #     del self.evaluator, self.ref_acts, self.ref_stats, self.ref_stats_spatial
                if self.args.check_dm_performance:
                    if os.path.exists(os.path.join(os.path.join(self.args.dm_sample_path_seed_42, 'single_npz'), f'stats')):
                        with open(os.path.join(os.path.join(self.args.dm_sample_path_seed_42, 'single_npz'), f'acts'), 'rb') as f:
                            sample_acts = pickle.load(f)
                            sample_acts = (sample_acts['acts'], sample_acts['acts_spatial'])
                        with open(os.path.join(os.path.join(self.args.dm_sample_path_seed_42, 'single_npz'), f'stats'), 'rb') as f:
                            sample_stats = pickle.load(f)
                            sample_stats, sample_stats_spatial = (sample_stats['stats'], sample_stats['stats_spatial'])
                    else:
                        sample_acts, sample_stats, sample_stats_spatial = self.calculate_inception_stats(self.args.data_name,
                                                                        self.args.dm_sample_path_seed_42,
                                                                        num_samples=self.args.eval_num_samples)
                    logger.log("Inception Score-50k:", self.evaluator.compute_inception_score(sample_acts[0]))
                    logger.log("FID-50k:", sample_stats.frechet_distance(self.ref_stats))
                    logger.log("sFID-50k:", sample_stats_spatial.frechet_distance(self.ref_stats_spatial))
                    prec, recall = self.evaluator.compute_prec_recall(self.ref_acts[0], sample_acts[0])
                    logger.log("Precision:", prec)
                    logger.log("Recall:", recall)
                    if self.args.gpu_usage:
                        self.print_gpu_usage('After computing DM FIDs')
                    #self.evaluator.sess.close()
                    if self.args.eval_fid:
                        tf.reset_default_graph()
            gc.collect()
            th.cuda.empty_cache()
            if self.args.eval_fid:
                tf.disable_eager_execution()

            if self.args.gpu_usage:
                self.print_gpu_usage('After emptying cache')


    def get_batch(self):
        if self.args.training_mode == 'pgd':
            if self.step % 2 == 0:
                batch, cond = next(self.data)
            else:
                batch, cond = next(self.data_for_GAN)
        else:
            batch, cond = next(self.data)
        return batch, cond

    def run_loop(self):
        if self.args.gpu_usage:
            self.print_gpu_usage('Before training')
        saved = False
        while (
            self.step < self.args.lr_anneal_steps
            or self.global_step < self.total_training_steps
        ):
            batch, cond = self.get_batch()
            if self.args.large_log:
                print("batch size: ", batch.shape)
                print("rank: ", dist.get_rank())
            if self.args.intermediate_samples:
                if self.step > self.initial_step + 1 and (self.step % self.args.sample_interval == 0):
                    self.sampling(model=self.ddp_decoder, sampler='onestep' if self.args.training_mode == 'pgd' else 'heun',
                                  step=1 if self.args.training_mode == 'pgd' else self.args.sampling_steps, resize=False,
                                  num_samples=self.args.sampling_batch, png=True,
                                  sample_dir=f'{self.step}_{"decoder" if self.args.training_mode == "pgd" else f"ode_heun_{self.args.sampling_steps}"}', random_class=False)
                    model_state_dict = copy.deepcopy(self.decoder.state_dict())
                    for rate, params in zip(self.ema_rate, self.ema_params):
                        state_dict = self.mp_decoder_trainer.master_params_to_state_dict(params)
                        self.decoder.load_state_dict(state_dict, strict=False)
                        self.sampling(model=self.decoder, sampler='onestep' if self.args.training_mode == 'pgd' else 'heun',
                                      step=1 if self.args.training_mode == 'pgd' else self.args.sampling_steps, resize=False,
                                      num_samples=self.args.sampling_batch, png=True,
                                      sample_dir=f'{self.step}_{rate}_ema_{"decoder" if self.args.training_mode == "pgd" else f"ode_heun_{self.args.sampling_steps}"}', random_class=False)
                    self.decoder.load_state_dict(model_state_dict, strict=True)
                    del model_state_dict, state_dict

            if self.args.pretraining_step != -1 and self.global_step == self.args.pretraining_step:
                del self.mp_decoder_trainer, self.opt_dec, self.ddp_decoder
                self.args.resume_checkpoint = bf.join(get_blob_logdir(), f"decoder_{self.args.pretraining_step-1}")
                self.mp_decoder_trainer, self.opt_dec, self.ddp_decoder, = \
                    self.load_model(self.decoder, 'decoder', self.args.lr_dec, self.args.use_fp16, self.ema_rate)

            self.run_step(batch, cond)
            if self.args.gpu_usage:
                self.print_gpu_usage('Before training')
            if (
                self.global_step
                and self.args.eval_decoder_interval != -1
                and self.global_step % self.args.eval_decoder_interval == self.args.eval_decoder_interval - 1
                and self.global_step >= self.args.pretraining_step
                or self.step == self.args.lr_anneal_steps - 1
                or self.global_step == self.total_training_steps - 1
            ):
                if self.args.gpu_usage:
                    self.print_gpu_usage('Before emptying cache in evaluation 1')
                gc.collect()
                th.cuda.empty_cache()
                if self.args.gpu_usage:
                    self.print_gpu_usage('After emptying cache in evaluation 1')
                if dist.get_rank() == 0:
                    config = tf.ConfigProto(
                        allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
                    )
                    config.gpu_options.allow_growth = True
                    config.gpu_options.per_process_gpu_memory_fraction = 0.1
                    self.evaluator = Evaluator(tf.Session(config=config), batch_size=100)
                    self.ref_acts = self.evaluator.read_activations(self.args.ref_path)
                    self.ref_stats, self.ref_stats_spatial = self.evaluator.read_statistics(self.args.ref_path,
                                                                                            self.ref_acts)
                model_state_dict = copy.deepcopy(self.decoder.state_dict())
                #self.evaluation(model=self.decoder, step=1, rate=0.0)
                #logger.log('Evaluation with model parameter end')
                for rate, params in zip(self.ema_rate, self.ema_params):
                    if not self.args.compute_ema_fids:
                        if rate != 0.999:
                            continue
                    state_dict = self.mp_decoder_trainer.master_params_to_state_dict(params)
                    self.decoder.load_state_dict(state_dict, strict=False)
                    self.evaluation(model=self.decoder, step=1 if self.args.training_mode == 'pgd' else self.args.sampling_steps, rate=rate)
                    logger.log(f'Evaluation with {rate}-EMA model parameter end')
                    del state_dict
                self.decoder.load_state_dict(model_state_dict, strict=True)
                del model_state_dict
                if dist.get_rank() == 0:
                    self.evaluator.sess.close()
                    del self.evaluator.sess, self.evaluator.manifold_estimator, self.evaluator.image_input, self.evaluator.softmax_input
                    del self.evaluator.pool_features, self.evaluator.softmax
                    tf.reset_default_graph()
                    del self.evaluator, self.ref_acts, self.ref_stats, self.ref_stats_spatial
                gc.collect()
                th.cuda.empty_cache()
            dist.barrier()
            if (
                    self.global_step != -1
                    and self.global_step % self.args.eval_ode_interval == self.args.eval_ode_interval - 1
                    and self.global_step < self.args.pretraining_step
                    or self.global_step == self.args.pretraining_step - 1
            ):
                if self.args.gpu_usage:
                    self.print_gpu_usage('Before emptying cache in evaluation 1')
                gc.collect()
                th.cuda.empty_cache()
                if dist.get_rank() == 0:
                    config = tf.ConfigProto(
                        allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
                    )
                    config.gpu_options.allow_growth = True
                    config.gpu_options.per_process_gpu_memory_fraction = 0.05
                    self.evaluator = Evaluator(tf.Session(config=config), batch_size=50)
                    self.ref_acts = self.evaluator.read_activations(self.args.ref_path)
                    self.ref_stats, self.ref_stats_spatial = self.evaluator.read_statistics(self.args.ref_path, self.ref_acts)
                model_state_dict = copy.deepcopy(self.decoder.state_dict())
                #self.evaluation(model=self.decoder, step=1, rate=0.0)
                #logger.log('Evaluation with ODE parameter end')
                for rate, params in zip(self.ema_rate, self.ema_params):
                    if not self.args.compute_ema_fids:
                        if rate != 0.999:
                            continue
                    state_dict = self.mp_decoder_trainer.master_params_to_state_dict(params)
                    self.decoder.load_state_dict(state_dict, strict=False)
                    self.evaluation(model=self.decoder, step=1 if self.args.training_mode == 'pgd' else self.args.sampling_steps, rate=rate)
                    logger.log(f'Evaluation with {rate}-EMA model parameter end')
                    del state_dict
                self.decoder.load_state_dict(model_state_dict, strict=True)
                del model_state_dict
                if dist.get_rank() == 0:
                    self.evaluator.sess.close()
                    del self.evaluator.sess, self.evaluator.manifold_estimator, self.evaluator.image_input, self.evaluator.softmax_input
                    del self.evaluator.pool_features, self.evaluator.softmax
                    tf.reset_default_graph()
                    del self.evaluator, self.ref_acts, self.ref_stats, self.ref_stats_spatial
                logger.log(f'Evaluation with {self.args.ode_ema_rate}-EMA ODE parameter end')
                gc.collect()
                th.cuda.empty_cache()
            dist.barrier()

            saved = False
            if (
                self.global_step
                and self.args.save_interval != -1
                and self.global_step % self.args.save_interval == 0
                and self.global_step > self.args.pretraining_step
            ):
                self.save()
                saved = True
                gc.collect()
                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.global_step == self.args.pretraining_step - 1:
                self.save(only_model=True)
                gc.collect()
                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.global_step % self.args.log_interval == 0:
                logger.dumpkvs()
                logger.log(datetime.datetime.now().strftime("SONY-%Y-%m-%d-%H-%M-%S"))

        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()

    def save_check(self, rate):
        if self.args.training_mode.lower() == 'ctm':
            assert rate == 0.999
            fid = self.eval(step=1, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                                  class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                          metric='similarity', delete=True, out=True)
            return fid

    def evaluation(self, model, step, rate):
        self.eval(model=model, step=step, sampler='onestep' if step == 1 else 'heun', rate=rate, ctm=False, delete=True)

    def run_step(self, batch, cond):
        #current = time.time()
        self.forward_backward(batch, cond)
        if self.args.training_mode == 'pgd':
            if self.step % 2 == 0:
                self.update_parameter(self.mp_decoder_trainer, self.opt_dec, self.ema_rate)
                if self.args.separate_update and self.step >= self.args.pretraining_step:
                    self.forward_backward(batch, cond, 'gan')
                    self.update_parameter(self.mp_decoder_trainer, self.opt_dec, self.ema_rate)
                if self.args.recon_discriminator:
                    self.forward_backward(batch, cond, 'discriminator')
                    self.mp_recon_discriminator_trainer.optimize(self.opt_recon_disc)
            if self.step % 2 == 1:
                if self.step >= self.args.pretraining_step:
                    self.mp_decoder_discriminator_trainer.optimize(self.opt_decoder_disc)
        else:
            self.update_parameter(self.mp_decoder_trainer, self.opt_dec, self.ema_rate)

        self.step += 1
        self.global_step += 1
        self.log_step()

    def loss_compute(self, ddp1, ddp2, compute_losses):
        losses = {}
        if ddp1 == None:
            losses = compute_losses()
        else:
            with ddp1.no_sync():
                if ddp2 == None:
                    losses = compute_losses()
                else:
                    with ddp2.no_sync():
                        losses = compute_losses()
        return losses

    def forward_backward(self, batch, cond, mode='reconstruction'):
        self.mp_decoder_trainer.zero_grad()
        if self.args.recon_discriminator:
            self.mp_recon_discriminator_trainer.zero_grad()
        if self.args.decoder_discriminator_training:
            self.mp_decoder_discriminator_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]

            if self.args.training_mode == 'pgd':
                compute_losses = functools.partial(
                    self.diffusion.pgd_losses,
                    step=self.step,
                    encoder=self.ode,
                    decoder=self.ddp_decoder,
                    x_start=micro,
                    model_kwargs=micro_cond,
                    recon_discriminator=self.ddp_recon_discriminator,
                    decoder_discriminator=self.ddp_decoder_discriminator,
                    init_step=self.initial_step,
                    loss1_grad_norm=self.loss1_grad_norm,
                    mode=mode,
                )
                if last_batch or not self.use_ddp:
                    losses = self.loss_compute(None, None, compute_losses)
                else:
                    if self.step % 2 == 0:
                        losses = self.loss_compute(self.ddp_decoder, self.ddp_recon_discriminator, compute_losses)
                    if self.step % 2 == 1:
                        losses = self.loss_compute(self.ddp_decoder_discriminator, None, compute_losses)
                if self.step % 2 == 0:
                    if self.args.decoder_training:
                        if mode == 'discriminator':
                            loss = losses['recon_discriminator_loss'].mean()
                            self.mp_recon_discriminator_trainer.backward(loss)
                        else:
                            if mode == 'reconstruction':
                                loss = losses['decoder_recon_loss'].mean()
                                if self.args.separate_update:
                                    if 'loss1_grad_norm' in list(losses.keys()):
                                        self.loss1_grad_norm = losses['loss1_grad_norm']
                                        del losses['loss1_grad_norm']
                                    else:
                                        self.loss1_grad_norm = th.norm(th.autograd.grad(loss, self.diffusion.get_last_layer(self.ddp_decoder, self.args.data_name), retain_graph=True)[0])
                                else:
                                    loss = loss + self.args.discriminator_weight * losses['discriminator_loss'].mean()
                                if self.args.recon_discriminator:
                                    loss = loss + self.args.recon_discriminator_weight * losses['recon_discriminator_loss'].mean()
                            elif mode == 'gan':
                                loss = self.args.discriminator_weight * losses['discriminator_loss'].mean()
                            else:
                                raise NotImplementedError
                            self.mp_decoder_trainer.backward(loss)
                        log_loss_dict({k: v.view(-1) for k, v in losses.items()})# if not v.mean().item()})
                if self.step % 2 == 1:
                    loss = losses['discriminator_loss'].mean()
                    #loss = self.diffusion.null(micro).mean()
                    log_loss_dict({k: v.view(-1) for k, v in losses.items()})
                    self.mp_decoder_discriminator_trainer.backward(loss)
            elif self.args.training_mode == 'diffusion':
                compute_losses = functools.partial(
                    self.diffusion.diffusion_losses,
                    step=self.step,
                    denoiser=self.ddp_decoder,
                    x_start=micro,
                    model_kwargs=micro_cond,
                )
                if last_batch or not self.use_ddp:
                    losses = self.loss_compute(None, None, compute_losses)
                else:
                    losses = self.loss_compute(self.ddp_decoder, None, compute_losses)
                loss = losses['diffusion_loss'].mean()
                self.mp_decoder_trainer.backward(loss)
                log_loss_dict({k: v.view(-1) for k, v in losses.items()})
            elif self.args.training_mode == 'pgd_encoder':
                batch_eval, cond_eval = next(self.data_for_GAN)
                batch_eval = batch_eval.to(dist_util.dev())
                micro_cond_eval = {k: v.to(dist_util.dev()) for k, v in cond_eval.items()}
                compute_losses = functools.partial(
                    self.diffusion.encoder_losses,
                    step=self.step,
                    encoder=self.ddp_decoder,
                    decoder=self.ode,
                    x_start=micro,
                    model_kwargs=micro_cond,
                    x_start_eval=batch_eval,
                    model_kwargs_eval=micro_cond_eval,
                )
                if last_batch or not self.use_ddp:
                    losses = self.loss_compute(None, None, compute_losses)
                else:
                    losses = self.loss_compute(self.ddp_decoder, None, compute_losses)
                loss = losses['encoder_loss'].mean()
                self.mp_decoder_trainer.backward(loss)
                log_loss_dict({k: v.view(-1) for k, v in losses.items()})

    @th.no_grad()
    def eval(self, model, step=1, sampler='exact', teacher=False, ctm=False, rate=0.999, generator=None, class_generator=None, metric='fid', delete=False, out=False):
        if not model:
            model = self.decoder
        sample_dir = f"{self.step}_{sampler}_{step}_{rate}"
        if generator != None:
            sample_dir = sample_dir + "_seed_42"
        self.sampling(model=model, sampler=sampler, teacher=teacher, step=step,
                      num_samples=self.args.eval_num_samples, batch_size=self.args.eval_batch,
                      rate=rate, ctm=ctm, png=False, resize=False, generator=generator,
                      class_generator=class_generator, sample_dir=sample_dir, fid_eval=True)
        gc.collect()
        th.cuda.empty_cache()
        if dist.get_rank() == 0:
            if self.args.data_name.lower() == 'cifar10':
                mu, sigma = self.calculate_inception_stats(self.args.data_name,
                                                           os.path.join(get_blob_logdir(), sample_dir),
                                                           num_samples=self.args.eval_num_samples)
                logger.log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate}"
                           f" FID-{self.args.eval_num_samples // 1000}k: {self.compute_fid(mu, sigma)}")
                if delete:
                    shutil.rmtree(os.path.join(get_blob_logdir(), sample_dir))
                if out:
                    return self.compute_fid(mu, sigma)
            else:
                sample_acts, sample_stats, sample_stats_spatial = self.calculate_inception_stats(self.args.data_name,
                                                                             bf.join(get_blob_logdir(), sample_dir),
                                                                             num_samples=self.args.eval_num_samples)
                logger.log(f"Inception Score-{self.args.eval_num_samples // 1000}k:", self.evaluator.compute_inception_score(sample_acts[0]))
                logger.log(f"FID-{self.args.eval_num_samples // 1000}k:", sample_stats.frechet_distance(self.ref_stats))
                logger.log(f"sFID-{self.args.eval_num_samples // 1000}k:", sample_stats_spatial.frechet_distance(self.ref_stats_spatial))
                prec, recall = self.evaluator.compute_prec_recall(self.ref_acts[0], sample_acts[0])
                logger.log("Precision:", prec)
                logger.log("Recall:", recall)
                del sample_acts, sample_stats, sample_stats_spatial
                if delete:
                    shutil.rmtree(os.path.join(get_blob_logdir(), sample_dir))
                #self.evaluator.sess.close()
                #tf.reset_default_graph()



    def log_step(self):
        step = self.global_step
        logger.logkv("step", step)
        logger.logkv("samples", (step + 1) * self.global_batch)


def parse_resume_step_from_filename(filename, splitter='decoder'):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split(f"{splitter}_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, type_, rate):
    if main_checkpoint is None:
        return None
    #if rate:
    filename = f"{type_}_ema_{rate}_{(step):06d}.pt"
    #else:
    #    filename = f"{type_}_ema_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(f"{key} mean", values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        logger.logkv_mean(f"{key} std", values.std().item())
        #for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #    quartile = int(4 * sub_t / diffusion.num_timesteps)
        #    logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
