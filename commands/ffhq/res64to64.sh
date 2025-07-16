DEFAULT_FLAGS="--xz_type=npz --map_location=cuda --use_MPI=True --new=False --decoder_gan_frequency=1 --num_workers=16 --pretraining_step=-1 --resblock_updown=True --use_scale_shift_norm=True --eval_ode_interval=5000000000 --num_heun_step_random=False --dev_log=True --heun_step_strategy=weighted --save_png=True --check_dm_performance=False --intermediate_samples=True --eval_fid=True --class_cond=True --load_ode=False --diffusion_schedule_sampler=lognormal --p_mean=-1.2 --p_std=1.2 --sigma_max=80. --rho=7"
IMPORTANT_FLAGS="--decoder_style=unet --loss_norm=cnn_vit --ema_rate=0.999 --training_mode=heun"
DECODER_FLAGS="--pretraining_step=-1 --activate_from=12  --num_channels=192 --num_head_channels=64 --num_res_blocks=3"
LOG_FLAGS="--gpu_usage=False --large_log=False" # --train_classes=-2"
# CKPT_FLAGS="--out_dir YOUR_OUT_DIR --ref_path FID_STATS_PATH --teacher_model_path STAGE1_PRETRAINED_DM_PATH --data_dir ImageNet_DATA_DIR --z_no_flip_dir DATA_LATENT_PAIR_DIR --z_flip_dir FLIPED_DATA_LATENT_PAIR_DIR"

RUNTIME_FLAGS="--device=0 \
               --port 128 \
               --use_MPI=True \
               --use_fp16=False \
               --separate_update=False \
               --recon_discriminator=False \
               --recon_discriminator_weight=0.2 \
               --decoder_adaptive_weight=True \
               --decoder_override=False \
               --superres=False \
               --progressive=False \
               --decoder_training=True \
               --decoder_discriminator_training=True \
               --discriminator_weight=0.2"

BATCH_FLAGS="--sampling_batch=8 \
             --microbatch 96 \
             --global_batch_size=96 \
             --eval_batch=512"

SIZE_FLAGS="--image_size=64 \
            --input_size=64 \
            --pretrained_input_size=64 \
            --pretrained_output_size=64"

INTERVAL_FLAGES="--eval_decoder_interval=10000 \
                 --save_interval=50000 \
                 --eval_num_samples=50000 \
                 --sample_interval=10000 \
                 --save_period=10000 \
                 --log_interval=10000"

CKPT_FLAGS="--out_dir out/ffhq_stage2 \
            --ref_path models/evaluation/ffhq_ref_batch.npz \
            --teacher_model_path models/edm-ffhq-64x64-uncond-vp.pkl \
            --data_dir /home/bratosiewicz/data/ffhq/train \
            --z_no_flip_dir /home/bratosiewicz/data/ffhq-xz-no-flip/edm_heun_sampler_40_steps_ond-vp_itrs_model_ema_7_rho \
            --z_flip_dir /home/bratosiewicz/data/ffhq-xz-flip/edm_heun_sampler_40_steps_ond-vp_itrs_model_ema_7_rho"

python3 train.py $DEFAULT_FLAGS $IMPORTANT_FLAGS $DECODER_FLAGS $LOG_FLAGS $RUNTIME_FLAGS $BATCH_FLAGS $SIZE_FLAGS $INTERVAL_FLAGES $CKPT_FLAGS