export PROJECT=$PROJECT_NAME
WEIGHT_PATH="/home/opensora/shebin/pre_weights/"
env
export WANDB_MODE='offline'

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example_on_npu.yaml \
    --machine_rank=${MACHINE_RANK} \
    --main_process_ip=${MAIN_PROCESS_IP_VALUE} \
    opensora/train/train_t2v.py \
    --model OpenSoraT2V-B/122 \
    --text_encoder_name ${WEIGHT_PATH}/DeepFloyd/t5-v1_1-xxl \
    --cache_dir "../cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "${WEIGHT_PATH}/CausalVAEModel_4x8x8_0430/" \
    --video_data "./scripts/train_data/video_data_on_npu.txt" \
    --image_data "./scripts/train_data/image_data_on_npu.txt" \
    --sample_rate 1 \
    --num_frames 1 \
    --max_height 256 \
    --max_width 256 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 0.5 \
    --interpolation_scale_w 0.5 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=8 \
    --dataloader_num_workers 20 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --output_dir="/home/image_data/checkpoints/${PROJECT}/" \
    --allow_tf32 \
    --model_max_length 512 \
    --use_image_num 0 \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --guidance_scale 2.5 \
    --resume_from_checkpoint="latest"