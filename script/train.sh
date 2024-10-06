#!/bin/sh
NGPUS=${NGPUS:-4}
NNODES=${NNODES:-1}
max_steps=$((10000 / NNODES))
torchrun --nproc_per_node $NGPUS --nnodes $NNODES \
        train.py \
        --use_flash_attention_2 flash \
        --deepspeed "config/ds_configs/stage2.json" \
        --ddp_timeout 18000 \
        --dataset_cache_dir ../data/c4 \
        --output_dir "output/${TYPE}_small_c4" \
        --config_name "config/${TYPE}.json" \
        --resume_from_checkpoint true \
        --max_steps $max_steps \
        --warmup_ratio 0.02 \
        --lr_scheduler_type polynomial \
        --save_steps 100 \
        --save_total_limit 1 \
        --logging_steps 50 \
        --weight_decay 0.01 \
        --learning_rate 1e-4 \
        --model_max_position_embeddings 1024 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16 \
        --do_train True \
        --do_predict True \
        --save_strategy "steps" \
        --gradient_checkpointing False \
        --bf16 True
