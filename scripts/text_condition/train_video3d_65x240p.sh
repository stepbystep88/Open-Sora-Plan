export WANDB_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
# export WANDB_MODE="offline"
export ENTITY="linbin"
export PROJECT="test3d_2node_65x240p_bs4_snr5_ema0_pt"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# NCCL setting
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=3
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1

accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example1.yaml \
    opensora/train/train_t2v.py \
    --model OpenSoraT2V-S/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "./cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/CausalVAEModel_4x8x8/" \
    --video_data "scripts/train_data/video_data_debug.txt" \
    --image_data "scripts/train_data/image_data_debug.txt" \
    --sample_rate 1 \
    --num_frames 65 \
    --max_height 240 \
    --max_width 320 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 0.5 \
    --interpolation_scale_w 0.5 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=4 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=100 \
    --output_dir="test3d_2node_65x240p_bs4_snr5_ema0_pt" \
    --allow_tf32 \
    --use_deepspeed \
    --model_max_length 512 \
    --use_image_num 0 \
    --enable_tiling \
    --enable_tracker \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --pretrained PixArt-Alpha-XL-2-512.safetensors
