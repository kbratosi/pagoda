DEFAULT_FLAGS="--xz_type=npz --map_location=cuda --use_MPI=True --new=False --decoder_gan_frequency=1 --num_workers=16 --pretraining_step=-1 --resblock_updown=True --use_scale_shift_norm=True --eval_ode_interval=5000000000 --num_heun_step_random=False --dev_log=True --heun_step_strategy=weighted --save_png=True --check_dm_performance=False --intermediate_samples=True --eval_fid=True --load_ode=False --p_mean=-1.2 --p_std=1.2 --sigma_max=80. --rho=7"
IMPORTANT_FLAGS="--decoder_style=unet --loss_norm=lpips --training_mode=pgd"
DECODER_FLAGS="--pretraining_step=-1 --activate_from=12"
STAGE_1_FLAGS="--num_channels=192 --num_channels_high=256 --num_head_channels=64 --num_res_blocks=3 --diffusion_weight_schedule=scaled_karras_weight --diffusion_schedule_sampler=lognormal --ema_rate=0.9999"
LOG_FLAGS="--gpu_usage=False --large_log=False" # --train_classes=-2"
# CKPT_FLAGS="--out_dir YOUR_OUT_DIR --ref_path FID_STATS_PATH --teacher_model_path STAGE1_PRETRAINED_DM_PATH --data_dir ImageNet_DATA_DIR --z_no_flip_dir DATA_LATENT_PAIR_DIR --z_flip_dir FLIPED_DATA_LATENT_PAIR_DIR"

RUNTIME_FLAGS="--device=1 \
               --port 128 \
               --use_MPI=True \
               --use_fp16=True \
               --class_cond=False \
               --separate_update=False \
               --recon_discriminator=True \
               --recon_discriminator_weight=0.2 \
               --decoder_adaptive_weight=True \
               --decoder_override=True \
               --superres=False \
               --progressive=False \
               --decoder_training=True \
               --decoder_discriminator_training=True \
               --discriminator_weight=0.2"

BATCH_FLAGS="--sampling_batch=16 \
             --microbatch=16 \
             --global_batch_size=128 \
             --eval_batch=512 \
             --eval_num_samples=50000"

SIZE_FLAGS="--image_size=64 \
            --input_size=64 \
            --pretrained_input_size=64 \
            --pretrained_output_size=64"

INTERVAL_FLAGS="--eval_decoder_interval=2000 \
                --save_interval=2000 \
                --sample_interval=2000 \
                --save_period=2000 \
                --log_interval=100"

CKPT_FLAGS="--out_dir out/11-03-pgd-ffhq-stage-2 \
            --ref_path models/evaluation/ffhq_ref_batch.npz \
            --teacher_model_path /home/bratosiewicz/pagoda/models/mine/stage1_cm_ffhq_ema_0.9999_060000.pt \
            --data_dir /home/bratosiewicz/data/ffhq/train \
            --z_no_flip_dir /home/bratosiewicz/pagoda/out/ffhq-xz-no-flip \
            --z_flip_dir    /home/bratosiewicz/pagoda/out/ffhq-xz-flip"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/bratosiewicz/environments/pagoda/bin/python train.py $DEFAULT_FLAGS $IMPORTANT_FLAGS $DECODER_FLAGS $STAGE_1_FLAGS $LOG_FLAGS $RUNTIME_FLAGS $BATCH_FLAGS $SIZE_FLAGS $INTERVAL_FLAGS $CKPT_FLAGS