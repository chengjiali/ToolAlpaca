#!/bin/bash

declare -a cat=('Development' 'Geocoding' 'Government' 'Transportation' 'Games & Comics')

export WANDB_MODE=disabled

for c in "${cat[@]}"
do
    CUDA_VISIBLE_DEVICES=0,2,3,4 accelerate launch train.py \
        --remove_class "${c}" \
        --output_dir test \
        --model_name_or_path lmsys/vicuna-7b-v1.1  \
        --data_path data/class_unlearn-"${c}".json \
        --dataloader_num_workers 16 \
        --bf16 True \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 10 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --lazy_preprocess False

done