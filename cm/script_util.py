import argparse
import numpy as np
import torch

from cm.karras_diffusion import KarrasDenoiser
from cm.resample import create_named_schedule_sampler

import blobfile as bf
import os
from torchvision.utils import make_grid, save_image

def ctm_data_defaults(data_name):
    return dict(
        train_classes=-1,
        type='png',
        sigma_data=0.5,
        deterministic=False,
        num_classes=1000,
        flip_ratio=0.5,
    )

def ctm_loss_defaults(data_name):
    return dict(
        # CTM hyperparams
        ctm_training=True,
        consistency_weight=1.0,
        ctm_estimate_outer_type='target_model_sg',
        ctm_estimate_inner_type='model',
        ctm_target_inner_type='model_sg',
        ctm_target_matching=False,
        sample_s_strategy='uniform',
        heun_step_strategy='weighted',
        heun_step_multiplier=1.0,
        outer_parametrization='euler',
        inner_parametrization='edm',
        time_continuous=False,
        self_learn=False,
        self_learn_iterative=False,
        target_matching=False,

        # DSM hyperparams
        diffusion_training=True,
        apply_adaptive_weight=True,
        denoising_weight=1.,
        diffusion_mult = 0.7,
        diffusion_schedule_sampler='halflognormal',
        diffusion_training_frequency=1.,

        # GAN hyperparams
        d_lr=0.002,
        gan_specific_batch=False,
        gan_micro_batch=32,
        gan_real_free=True,
        discriminator_start_itr=0,
        discriminator_use_fp16=False,
        d_architecture='StyleGAN-XL',
        g_learning_period=1,
        gan_fake_outer_type='no',
        gan_fake_inner_type='',
        gan_real_inner_type='',
        gan_target_matching=False,
        data_augment=True,
        d_backbone=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'],
        d_apply_adaptive_weight=True,
        shift_ratio=0.125,
        cutout_ratio=0.2,
        gan_training_frequency=1.,
        gaussian_filter=False,
        blur_fade_itr=1000,
        blur_init_sigma=2,
        prob_aug=1.0,
        gan_different_augment=False,
        gan_num_heun_step=17 if data_name == 'cifar10' else 39,
        gan_heun_step_strategy='uniform',
        gan_specific_time=False,
        gan_low_res_train=False,
        d_opt_load=True,


        ######
        encoder_training=True,
        spherical_regularization_weight=0.0001,
        spherical_regularization=True,
        encoder_alpha=0.5,
        encoder_style='ldm',
        reconstruction_weight=0.5,
        kl_regularization_weight=0.1,
        encoder_ema_rate=0.999 if data_name == 'cifar10' else "0.999,0.9999,0.9999432189950708",
        std=0.0,
        encoder_discriminator_training=False,
        encoder_discriminator_initiate_itr=0,
        use_decoder_ema_for_encoder_train=True,

        decoder_training=True,
        decoder_alpha=0.5,
        decoder_fKL_weight=0.5,
        decoder_style='unet',
        decoder_ch=128,
        decoder_num_res_blocks=2,
        decoder_ch_mult="2,2",
        decoder_resolution=64,
        decoder_scale='karras',
        decoder_discriminator_training=False,
        decoder_discriminator_initiate_itr=0,
        ema_rate="0.999,0.9999" if data_name == 'cifar10' else "0.999,0.9999,0.9999432189950708",
        decoder_deactivate_alpha=True,
        decoder_distill_frequency=1,
        decoder_model_channels=128,
        decoder_channel_mult="2,2,2",
        decoder_num_blocks=4,
        decoder_distill_weight=0.1,
        eval_decoder_interval=1000,
        gan_major=True,
        use_encoder_ema_for_decoder_train=True,
        decoder_reverse=False,
        decoder_override=True,
        pretraining_step=-1,
        decoder_gan_frequency=5,

        ode_alpha=0.5,
        ode_scale='karras',
        ode_ema_rate=0.9999 if data_name == 'cifar10' else "0.999,0.9999,0.9999432189950708",
        eval_ode_interval=10000,
        alpha_discrete=False,
        use_encoder_ema_for_ode_train=True,
        load_ode=False,

        dev_log=False,

        diffusion_type='ve',
        start_heun_step=3,

        p_mean=-1.2,
        p_std=1.2,

        adaptive_weight=False,
        discriminator_adaptive_weight=True,
        discriminator_weight=0.1,

        stage_1_itr=1000,
        stage_2_itr=1000,
        use_fp16=True,
        eval_batch_size=1000,

        xz_data_dir='',
        xz_type='npz',
        num_xz_data=-1,
        decoder_model_path='',
        z_no_flip_dir='',
        z_flip_dir='',
        superres=False,
        lowerres=False,
        progressive=False,
        new=False,
        pretrained_input_size=64,
        pretrained_output_size=64,
        activate_from=11,
        num_channels_high=256,
        use_scale_shift_norm_high=False,
        num_res_blocks_high=2,
        separate_update=False,
        num_init_layers=3,
        recon_discriminator=False,
        recon_discriminator_weight=0.2,
        decoder_adaptive_weight=False,
        new_arch='no',
        wavelet_input=False,
        discriminator_class_cond=True,
        upsample_to_64=False,
        attention_type='flash',
        data_proportion=1.,
        load_encoder=False,
        encoder_model_path="",
    )

def ctm_train_defaults(data_name):
    return dict(
        beta_min=0.1,
        beta_max=20.,
        multiplier=1.,
        num_heun_step=17 if data_name == 'cifar10' else 39,
        num_heun_step_random=True,

        # Network architecture
        edm_nn_ncsn=False,
        edm_nn_ddpm=True if data_name == 'cifar10' else False,
        in_channels=3,
        linear_probing=False,
        target_subtract=False,
    )

def ctm_eval_defaults(data_name):
    return dict(
        intermediate_samples=False,
        sampling_batch=64,
        sample_interval=1000 if data_name == 'cifar10' else 1000,
        sampling_steps=18 if data_name == 'cifar10' else 40,
        eval_interval=1000,
        eval_num_samples=50000,
        eval_batch=500,
        #ref_path='/home/dongjun/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz' if data_name == 'cifar10' else "",
        ref_path='/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz' if data_name == 'cifar10' \
            else "/home/fp084243/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz",
        ref_feat_path='',
        large_log=False,
        compute_ema_fids=True,
        #dm_sample_path_seed_42='/data2/dongjun/EighthArticleExperimentalResults/CIFAR10/DM/EDM-VP/fp16-seed-42/edm_heun_sampler_18_steps_ond-vp_itrs_model_ema' if data_name == 'cifar10' else "",
        dm_sample_path_seed_42='/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/DM/heun_18_seed_42_ver2' if data_name == 'cifar10' else "",
        ae_image_path_seed_42='',
        eval_seed=42,
        eval_fid=True,
        eval_similarity=True,
        save_png=False,
        check_ctm_denoising_ability=False,
        check_dm_performance=True,
        sanity_check=False,
        save_period=1000 if data_name == 'cifar10' else 1000,
        clip_denoised=False,
        clip_output=True,
        gpu_usage=False,
        eval_large_nfe=True,
    )

def cm_train_defaults(data_name):
    return dict(
        #teacher_model_path="/home/dongjun/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-uncond-vp.pkl" if data_name == 'cifar10' else "",
        teacher_model_path="" if data_name == 'cifar10' else "",
        teacher_dropout=0.0 if data_name == 'cifar10' else 0.1,
        training_mode="ctm",
        target_ema_mode="fixed",
        scale_mode="fixed",
        total_training_steps=600000,
        start_ema=0.999,
        start_scales=18 if data_name == 'cifar10' else 40,
        end_scales=18 if data_name == 'cifar10' else 40,
        distill_steps_per_iter=50000,
        loss_norm="lpips",
        port=6,
    )

def model_and_diffusion_defaults(data_name):
    """
    Defaults for image training.
    """
    res = dict(
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7,
        image_size=32 if data_name == 'cifar10' else 64,
        num_channels=192,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="",
        encoder_channel_mult="",
        decoder_channel_mult_="",
        input_size=64,
        dropout=0.0,
        class_cond=False if data_name == 'cifar10' else True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        learn_sigma=False,
        weight_schedule="uniform",
        weight_schedule_multiplier=1.,
        diffusion_weight_schedule="scaled_karras_weight",
        rescaling=False,
    )
    return res

def train_defaults(data_name):
    """
        Defaults for model training.
    """
    res = dict(
        out_dir="",
        #data_dir="/home/dongjun/EighthArticleExperimentalResults/CIFAR10/train" if data_name == 'cifar10' else "",
        data_dir="/home/acf15618av/dataset/CIFAR10/train" if data_name == 'cifar10' else "",
        schedule_sampler="uniform_decoder",
        lr_dec=0.0004 if data_name == 'cifar10' else 0.000008,
        lr_enc=0.0004 if data_name == 'cifar10' else 0.000008,
        lr_ode=0.001 if data_name == 'cifar10' else 0.0001,
        lr_disc=0.002,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=128 if data_name == 'cifar10' else 2048,
        batch_size=-1,
        microbatch=64 if data_name.lower() == 'cifar10' else -1,  # -1 disables microbatches
        # comma-separated list of EMA values
        log_interval=1000,
        save_interval=1000000,
        save_check_period=10000000,
        resume_checkpoint="",
        fp16_scale_growth=1e-3,
        device_id=0,
        num_workers=4,
        use_MPI=False,
        map_location='cpu',
    )
    return res

def save(x, save_dir, name, npz=False):
    nrow = int(np.sqrt(x.shape[0]))
    image_grid = make_grid((x + 1.) / 2., nrow, padding=2)
    with bf.BlobFile(os.path.join(save_dir, f"{name}.png"), "wb") as fout:
        save_image(image_grid, fout)
    if npz:
        sample = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.cpu().detach()
        os.makedirs(os.path.join(save_dir, 'targets'), exist_ok=True)
        np.savez(os.path.join(save_dir, f"targets/{name}.npz"), sample.numpy())

def create_model_and_diffusion(args, feature_extractor=None, discriminator_feature_extractor=None, teacher=False, type_='decoder'):
    schedule_sampler = create_named_schedule_sampler(args, args.schedule_sampler, args.start_scales)
    diffusion_schedule_sampler = create_named_schedule_sampler(args, args.diffusion_schedule_sampler, args.start_scales)

    if args.data_name.lower() == 'cifar10':

        if type_ == 'ode':
            from cm.networks import EDMPrecond_CTM
            model = EDMPrecond_CTM(args, img_resolution=args.image_size, img_channels=3,
                                   label_dim=1000 if args.data_name.lower() == 'imagenet64' else 10 if args.class_cond else 0,
                                   use_fp16=args.use_fp16,
                                   sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                                   sigma_data=args.sigma_data,
                                   model_type='SongUNet' if args.data_name.lower() == 'cifar10' else 'DhariwalUNet',
                                   teacher=teacher, teacher_model_path=args.teacher_model_path,
                                   training_mode=args.training_mode,
                                   arch='ddpmpp' if args.data_name.lower() == 'cifar10' else 'adm',
                                   linear_probing=args.linear_probing, encoder=(type_ == 'encoder'))
        else:
            if args.decoder_style == 'ldm':
                from cm.ldm_encoder import Decoder
                model = Decoder(ch=args.decoder_ch, out_ch=3, num_res_blocks=args.decoder_num_res_blocks, ch_mult=args.decoder_ch_mult,
                                attn_resolutions=[], dropout=0.0, resamp_with_conv=True, in_channels=3,
                                resolution=args.decoder_resolution, z_channels=3, double_z=True, use_linear_attn=False,
                                attn_type="vanilla", )
            elif args.decoder_style == 'stylegan':
                from cm.networks_stylegan3_resetting import Generator
                model = Generator(z_dim=int(args.image_size * args.image_size * 3),
                                  z_mid_dim=64,
                                  c_dim=64 if args.class_cond else 0,
                                  w_dim=512,
                                  img_resolution=args.image_size,
                                  img_channels=3,
                                  )
            else:
                from cm.networks import EDMPrecond_CTM
                model = EDMPrecond_CTM(args, img_resolution=args.image_size, img_channels=3,
                                       label_dim=1000 if args.data_name.lower() == 'imagenet64' else 10 if args.class_cond else 0,
                                       use_fp16=args.use_fp16,
                                       sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                                       sigma_data=args.sigma_data,
                                       model_type='SongUNet' if args.data_name.lower() == 'cifar10' else 'DhariwalUNet',
                                       teacher=teacher, teacher_model_path=args.teacher_model_path or args.model_path,
                                       training_mode=args.training_mode,
                                       arch='ddpmpp' if args.data_name.lower() == 'cifar10' else 'adm',
                                       linear_probing=args.linear_probing, encoder=(type_ == 'encoder'),
                                       model_channels=args.decoder_model_channels, channel_mult=list(int(ch_mult) for ch_mult in args.decoder_channel_mult.split(",")),
                                       num_blocks=args.decoder_num_blocks)
    else:
        if args.decoder_style == 'unet':
            if type_ == 'encoder':
                model = create_model(
                    args,
                    input_size=32,
                    output_size=32,
                    num_channels_high=-1,
                    use_scale_shift_norm_high=-1,
                    num_res_blocks_high=-1,
                    encoder_channel_mult="",
                    decoder_channel_mult="",
                    num_channels=256,
                    num_res_blocks=3,
                    learn_sigma=False,
                    class_cond=args.class_cond,
                    use_checkpoint=args.use_checkpoint,
                    attention_resolutions="16,8",
                    num_heads=args.num_heads,
                    num_head_channels=args.num_head_channels,
                    num_heads_upsample=args.num_heads_upsample,
                    use_scale_shift_norm=args.use_scale_shift_norm,
                    dropout=args.dropout,
                    resblock_updown=args.resblock_updown,
                    use_fp16=args.use_fp16,
                    use_new_attention_order=args.use_new_attention_order,
                    training_mode=('teacher' if teacher else args.training_mode),
                    type_=type_,
                    new_arch=args.new_arch,
                    attention_type=args.attention_type,
                    # new_arch='ode' if type_ == 'ode' else args.new_arch,
                )
            else:
                model = create_model(
                    args,
                    input_size=args.pretrained_input_size if type_ == 'ode' or args.upsample_to_64 else args.input_size,
                    output_size=args.pretrained_output_size if type_ == 'ode' else args.image_size,
                    num_channels_high=args.num_channels_high,
                    use_scale_shift_norm_high=args.use_scale_shift_norm_high,
                    num_res_blocks_high=args.num_res_blocks_high,
                    encoder_channel_mult=args.encoder_channel_mult,
                    decoder_channel_mult=args.decoder_channel_mult_,
                    num_channels=args.num_channels,
                    num_res_blocks=args.num_res_blocks,
                    learn_sigma=False,
                    class_cond=args.class_cond,
                    use_checkpoint=args.use_checkpoint,
                    attention_resolutions=args.attention_resolutions,
                    num_heads=args.num_heads,
                    num_head_channels=args.num_head_channels,
                    num_heads_upsample=args.num_heads_upsample,
                    use_scale_shift_norm=args.use_scale_shift_norm,
                    dropout=args.dropout,
                    resblock_updown=args.resblock_updown,
                    use_fp16=args.use_fp16,
                    use_new_attention_order=args.use_new_attention_order,
                    training_mode=('teacher' if teacher else args.training_mode),
                    type_=type_,
                    new_arch=args.new_arch,
                    attention_type=args.attention_type,
                    #new_arch='ode' if type_ == 'ode' else args.new_arch,
                )
        elif args.decoder_style == 'stylegan':
            from cm.networks_stylegan3_resetting import Generator, SuperresGenerator
            model = Generator(z_dim=int(args.image_size * args.image_size * 3),
                              z_mid_dim=64,
                              c_dim=64 if args.class_cond else 0,
                              w_dim=512,
                              num_init_layers=args.num_init_layers,
                              img_resolution=args.image_size,
                              img_channels=3,
                              #up_factor=args.image_size//16
                              )
        elif args.decoder_style == 'ldm':
            from cm.sd_model import Decoder
            decoder_ch_mult = tuple(int(ch_mult) for ch_mult in args.decoder_ch_mult.split(","))
            model = Decoder(ch=args.decoder_ch, out_ch=3, num_res_blocks=args.decoder_num_res_blocks,
                            ch_mult=decoder_ch_mult,
                            attn_resolutions=[], dropout=args.dropout, resamp_with_conv=True, in_channels=3,
                            resolution=args.image_size, num_classes=(args.num_classes if args.class_cond else None))
        else:
            raise NotImplementedError
    diffusion = KarrasDenoiser(
        args=args, schedule_sampler=schedule_sampler,
        diffusion_schedule_sampler=diffusion_schedule_sampler,
        feature_extractor=feature_extractor,
        discriminator_feature_extractor=discriminator_feature_extractor,
    )
    return model, diffusion

def create_model(
    args,
    input_size,
    output_size,
    num_channels_high,
    use_scale_shift_norm_high,
    num_res_blocks_high,
    encoder_channel_mult,
    decoder_channel_mult,
    #image_size,
    num_channels,
    num_res_blocks,
    #channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    training_mode='',
    type_='decoder',
    new_arch='ode',
    attention_type='flash',
):
    if encoder_channel_mult == "":
        if input_size == 512:
            encoder_channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif input_size == 256:
            encoder_channel_mult = (1, 1, 2, 2, 4, 4)
        elif input_size == 128:
            encoder_channel_mult = (1, 1, 2, 3, 4)
        elif input_size == 64:
            encoder_channel_mult = (1, 2, 3, 4)
            if args.new and type_ == 'decoder':
                if output_size == 256:
                    encoder_channel_mult = (1, 1, 1, 2, 3, 4)
                elif output_size == 128:
                    encoder_channel_mult = (1, 1, 2, 3, 4)
                else:
                    raise NotImplementedError
        elif input_size == 32:
            if new_arch == 'enc_dec':
                encoder_channel_mult = (2, 3, 4)
            else:
                encoder_channel_mult = (1, 2, 4)
        elif input_size == 16:
            encoder_channel_mult = (3, 4)
        elif input_size == 8:
            encoder_channel_mult = (4,)
        else:
            raise ValueError(f"unsupported image size: {input_size}")
    else:
        encoder_channel_mult = tuple(int(ch_mult) for ch_mult in encoder_channel_mult.split(","))
    print("encoder_channel_mult: ", input_size, encoder_channel_mult)
    if decoder_channel_mult == "":
        if output_size == 512:
            #decoder_channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            if input_size == 32:
                if new_arch == 'only_dec':
                    decoder_channel_mult = (0.5, 1, 1, 1, 2, 2, 4)
                elif new_arch == 'enc_dec':
                    decoder_channel_mult = (0.5, 1, 1, 1, 2, 3, 4)
                else:
                    decoder_channel_mult = (0.5, 1, 1, 1, 1, 2, 4)
            else:
                decoder_channel_mult = (0.5, 1, 1, 1, 1, 2, 4)
        elif output_size == 256:
            #decoder_channel_mult = (1, 1, 2, 2, 4, 4)
            if input_size == 32:
                if new_arch == 'only_dec':
                    decoder_channel_mult = (1, 1, 1, 2, 2, 4)
                elif new_arch == 'enc_dec':
                    decoder_channel_mult = (1, 1, 1, 2, 3, 4)
                else:
                    decoder_channel_mult = (1, 1, 1, 1, 2, 4)
            else:
                decoder_channel_mult = (1, 1, 1, 2, 3, 4)
        elif output_size == 128:
            if input_size == 32:
                if new_arch == 'only_dec':
                    decoder_channel_mult = (1, 1, 2, 2, 4)
                elif new_arch == 'enc_dec':
                    decoder_channel_mult = (1, 1, 2, 3, 4)
                else:
                    decoder_channel_mult = (1, 1, 1, 2, 4)
            else:
                decoder_channel_mult = (1, 1, 2, 3, 4)
        elif output_size == 64:
            if input_size == 32:
                if new_arch == 'only_dec':
                    decoder_channel_mult = (1, 2, 2, 4)
                elif new_arch == 'enc_dec':
                    decoder_channel_mult = (1, 2, 3, 4)
                else:
                    decoder_channel_mult = (1, 1, 2, 4)
            else:
                decoder_channel_mult = (1, 2, 3, 4)
        elif output_size == 32:
            decoder_channel_mult = (1, 2, 4)
        else:
            raise ValueError(f"unsupported image size: {output_size}")
    else:
        try:
            decoder_channel_mult = tuple(int(ch_mult) for ch_mult in decoder_channel_mult.split(","))
        except:
            decoder_channel_mult = tuple(float(ch_mult) for ch_mult in decoder_channel_mult.split(","))
    print("decoder_channel_mult: ", output_size, decoder_channel_mult)
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(output_size // int(res) if args.new and type_ == 'decoder' else input_size // int(res))
    #print(attention_ds)
    #if type_ == 'decoder':
    #    import sys
    #    sys.exit()
    from .unet import UNetModel
    return UNetModel(
        input_size=input_size,
        output_size=output_size,
        model_channels_high=num_channels_high,
        use_scale_shift_norm_high=use_scale_shift_norm_high,
        num_res_blocks_high=num_res_blocks_high,
        encoder_channel_mult=encoder_channel_mult,
        decoder_channel_mult=decoder_channel_mult,
        attention_resolutions=tuple(attention_ds),
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        #attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        #channel_mult=channel_mult,
        num_classes=(args.num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        training_mode=training_mode,
        new=args.new,
        no_grad=(args.image_size != args.input_size),
        attention_type=attention_type,
    )

def create_encoder(
    args,
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    from .unet import EncoderUNetModel
    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=int(image_size * image_size * 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist":
            distill_stage = step // distill_steps_per_iter
            scales = start_scales // (2**distill_stage)
            scales = np.maximum(scales, 2)

            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1),
                0,
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = 2 // (2**sub_stage)
            sub_scales = np.maximum(sub_scales, 1)

            scales = np.where(scales == 2, sub_scales, scales)

            target_ema = 1.0
        else:
            raise NotImplementedError

        return float(target_ema), int(scales)

    return ema_and_scales_fn


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
