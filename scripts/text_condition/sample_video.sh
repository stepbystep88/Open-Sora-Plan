export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python opensora/sample/sample_t2v.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.0.0 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_7.txt \
    --ae CausalVAEModel_4x8x8 \
    --version 65x512x512 \
    --save_img_path "./sample_videos/prompt_list_7" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 250 \
    --enable_tiling
