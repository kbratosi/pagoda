DEFAULT_FLAGS="
--xz_type=npz
--map_location=cuda
--use_MPI=True
--use_fp16=True
--loss_norm=lpips
--new=False
--decoder_style=unet 
--decoder_reverse=False 
--gan_major=True
--use_encoder_ema_for_decoder_train=True 
--decoder_distill_weight=1.0 
--decoder_distill_frequency=1 
--decoder_gan_frequency=1 
--decoder_fKL_weight=0.0 
--decoder_discriminator_initiate_itr=0 
--stage_2_itr=-1 
--pretrained_input_size=64 
--input_size=64 
--num_workers=16 
--sampling_batch=8 
--num_heun_step=17 
--num_channels=192 
--num_head_channels=64 
--num_res_blocks=3 
--resblock_updown=True 
--use_scale_shift_norm=True 
--ema_rate=0.999 
--training_mode=pgd 
--eval_ode_interval=5000000000 
--num_heun_step_random=False 
--dev_log=True 
--heun_step_strategy=weighted 
--start_heun_step=3 
--save_png=True 
--check_dm_performance=False 
--intermediate_samples=True 
--eval_fid=True 
--eval_similarity=False 
--class_cond=True 
--load_ode=False 
--diffusion_schedule_sampler=lognormal 
--alpha_discrete=False 
--use_encoder_ema_for_ode_train=True 
--p_mean=-1.2 
--p_std=1.2 
--sigma_max=80. 
--rho=7 
--diffusion_weight_schedule=karras_weight 
--diffusion_type=ve 
--ode_ema_rate=0.999 
--stage_1_itr=-1"
DECODER_FLAGS="
--pretraining_step=-1 
--activate_from=12 
--separate_update=False 
--recon_discriminator=False 
--recon_discriminator_weight=0.2 
--decoder_adaptive_weight=True 
--image_size=64 
--num_channels=192 
--num_head_channels=64 
--num_res_blocks=3 
--resblock_updown=True 
--use_scale_shift_norm=True 
--pretrained_output_size=64 
--decoder_override=True 
--superres=False 
--progressive=False 
--decoder_training=True 
--decoder_discriminator_training=True 
--discriminator_weight=0.2"
CHANGEABLE_FLAGS="
--device_id=0 
--port 128 
--microbatch 260 
--global_batch_size=260 
--eval_decoder_interval=1000 
--save_interval=10000 
--eval_batch=800 
--eval_num_samples=50000 
--sample_interval=1000 
--save_period=1000 
--log_interval=100"
CKPT_FLAGS="
--out_dir YOUR_OUT_DIR 
--ref_path FID_STATS_PATH 
--teacher_model_path STAGE2_PRETRAINED_DM_PATH 
--data_dir ImageNet_DATA_DIR 
--z_no_flip_dir DATA_LATENT_PAIR_DIR 
--z_flip_dir FLIPED_DATA_LATENT_PAIR_DIR"
python image_sample.py 
$DEFAULT_FLAGS 
$DEFAULT_FLAGS 
$ENCODER_FLAGS 
$DECODER_FLAGS 
$DEC_FLAGS 
$ODE_FLAGS 
$CHANGEABLE_FLAGS 
$CKPT_FLAGS 
--out_dir /data/NinthArticleExperimentalResults/ImageNet64/experiments/test_v2_samples_50k 
--model_path=/data/NinthArticleExperimentalResults/ImageNet64/experiments/test_v2/decoder_ema_0.999_180000.pt 
--training_mode=pgd 
--class_cond=True 
--batch_size=500 
--eval_num_samples=50000 
--stochastic_seed=True 
--save_format=npz 
--ind_1=5 
--ind_2=3 
--use_MPI=True 
--sampler=onestep 
--sampling_steps=1 
--device_id=0