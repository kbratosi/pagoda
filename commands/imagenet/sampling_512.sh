JUNK="--decoder_reverse=False "

RUNTIME_FLAGS="--device_id=0 --port 128 --xz_type=npz --map_location=cuda --use_MPI=True --use_fp16=True --save_png=True --large_log=True"
GENERATOR_FLAGS="--training_mode=pgd --class_cond=True --decoder_style=unet --loss_norm=cnn_vit --num_res_blocks_high=2 --num_res_blocks=3 --resblock_updown=True"

DEFAULT_FLAGS="--new=False --gan_major=True --use_encoder_ema_for_decoder_train=True --decoder_distill_weight=1.0 --decoder_distill_frequency=1 --decoder_gan_frequency=1 --decoder_fKL_weight=0.0 --decoder_discriminator_initiate_itr=0 --stage_2_itr=-1  --num_workers=16  --num_heun_step=17 --ema_rate=0.999 --eval_ode_interval=5000000000 --num_heun_step_random=False --dev_log=True --heun_step_strategy=weighted --start_heun_step=3 --check_dm_performance=False --intermediate_samples=True --eval_fid=True --eval_similarity=False --load_ode=False --diffusion_schedule_sampler=lognormal --alpha_discrete=False --use_encoder_ema_for_ode_train=True --p_mean=-1.2 --p_std=1.2 --sigma_max=80. --rho=7 --diffusion_weight_schedule=karras_weight --diffusion_type=ve --ode_ema_rate=0.999 --stage_1_itr=-1"
DECODER_FLAGS="--pretraining_step=-1 --activate_from=12 --separate_update=False --recon_discriminator=True --recon_discriminator_weight=0.2 --decoder_adaptive_weight=True --use_scale_shift_norm=True --decoder_override=False --superres=True --progressive=True --decoder_training=False --decoder_discriminator_training=False --discriminator_weight=0.2"

CHANGEABLE_FLAGS="--sampling_steps=1
                  --batch_size=50 \
                  --train_classes=1000 \
                  --eval_num_samples=50000 \
                  --eval_decoder_interval=1000 \
                  --sample_interval=1000 \
                  --save_interval=10000 \
                  --save_period=1000 \
                  --log_interval=100"

ACTUALLY_IMPORTANT_FLAGS="--pretrained_input_size=64 \
                          --pretrained_output_size=512 \
                          --input_size=64 \
                          --image_size=512 \
                          --num_channels=192 \
                          --num_channels_high=192 \
                          --num_head_channels=64"

CKPT_FLAGS="--ref_path models/evaluation/VIRTUAL_imagenet64_labeled.npz \
            --data_dir /data/bratosiewicz/ILSVRC/Data/CLS-LOC/train \
            --z_no_flip_dir /data/bratosiewicz/imagenet-xz-no-flip/edm_heun_sampler_40_steps_64_ema_itrs_model_ema_7_rho \
            --z_flip_dir /data/bratosiewicz/imagenet-xz-flip/edm_heun_sampler_40_steps_64_ema_itrs_model_ema_7_rho"

python image_sample.py \
$RUNTIME_FLAGS \
$GENERATOR_FLAGS \
$DEFAULT_FLAGS \
$DECODER_FLAGS \
$CHANGEABLE_FLAGS \
$ACTUALLY_IMPORTANT_FLAGS \
$CKPT_FLAGS \
--out_dir out/sampling/8-24-pgd-512-1000_class-100_each \
--model_path models/stage3_onestep_generator_64x64_to_512x512_ema_0.999_070000_fid_1.80.pt \
--eval_num_samples=50000 \
--stochastic_seed=True \
--save_format=png_each \
--ind_1=5 \
--ind_2=3 \
--sampler=onestep \
