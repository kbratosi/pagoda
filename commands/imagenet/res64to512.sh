DEFAULT_FLAGS="--xz_type=npz --map_location=cuda --use_MPI=True --use_fp16=True --loss_norm=cnn_vit --new=False --decoder_style=unet --decoder_reverse=False --gan_major=True --use_encoder_ema_for_decoder_train=True --decoder_distill_weight=1.0 --decoder_distill_frequency=1 --decoder_gan_frequency=1 --decoder_fKL_weight=0.0 --decoder_discriminator_initiate_itr=0 --stage_2_itr=-1 --num_workers=16 --pretraining_step=-1 --num_heun_step=17 --num_channels=192 --num_head_channels=64 --num_res_blocks=3 --resblock_updown=True --use_scale_shift_norm=True --ema_rate=0.999 --training_mode=pgd --eval_ode_interval=5000000000 --num_heun_step_random=False --dev_log=True --heun_step_strategy=weighted --start_heun_step=3 --save_png=True --check_dm_performance=False --intermediate_samples=True --eval_fid=True --eval_similarity=False --class_cond=True --load_ode=False --diffusion_schedule_sampler=lognormal --alpha_discrete=False --use_encoder_ema_for_ode_train=True --p_mean=-1.2 --p_std=1.2 --sigma_max=80. --rho=7 --diffusion_weight_schedule=karras_weight --diffusion_type=ve --ode_ema_rate=0.999 --stage_1_itr=-1"
DECODER_FLAGS="--activate_from=9 --separate_update=True --recon_discriminator=True --recon_discriminator_weight=0.5 --decoder_adaptive_weight=False --num_channels_high=192 --use_scale_shift_norm_high=False --num_res_blocks_high=2 --decoder_override=True --superres=True --progressive=True --decoder_training=True --decoder_discriminator_training=True --discriminator_weight=1.0"
CHANGEABLE_FLAGS="--device_id=0 --port 128 --microbatch 4 --sampling_batch=4 --global_batch_size=512 --eval_decoder_interval=5000 --save_interval=10000 --eval_batch=128 --eval_num_samples=50000 --sample_interval=100 --save_period=5000 --log_interval=5000"
RESOLUTION_FLAGS="--pretrained_input_size=64 \
                  --input_size=64 \
                  --pretrained_output_size=256 \
                  --image_size=512"
CKPT_FLAGS="--out_dir out/64to512_test \
            --ref_path models/evaluation/VIRTUAL_imagenet64_labeled.npz \
            --teacher_model_path models/stage3_onestep_generator_64x64_to_256x256_ema_0.999_240000_fid_1.56.pt \
            --data_dir /data/bratosiewicz/ILSVRC/Data/CLS-LOC/train \
            --z_no_flip_dir /data/bratosiewicz/imagenet-xz-no-flip/edm_heun_sampler_40_steps_64_ema_itrs_model_ema_7_rho \
            --z_flip_dir /data/bratosiewicz/imagenet-xz-flip/edm_heun_sampler_40_steps_64_ema_itrs_model_ema_7_rho"
python train.py $DEFAULT_FLAGS $DECODER_FLAGS $CHANGEABLE_FLAGS $RESOLUTION_FLAGS $CKPT_FLAGS