#!/bin/bash
# conda deactivate; conda activate diffusers

DATASET_PATH="/root/projects/Data/VPS07_Kevin_Williams/"
OUTPUT_PATH="ckpt/5b-t2v-lora/debug/"
# validation_prompt="SKS Kevin walking down the beach."

# keep it
CACHE_PATH="/root/projects/.cache/huggingface/models" # "~/.cache"
MODEL_PATH="THUDM/CogVideoX-5b"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \
  train_cogvideox_lora.py \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --validation_prompt "SKS Kevin walking down the beach." \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 100 \
  --seed 42 \
  --rank 128 \
  --lora_alpha 64 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 1 \
  --num_train_epochs 30 \
  --checkpointing_steps 1000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb