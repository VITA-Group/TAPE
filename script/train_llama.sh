#!/bin/sh
NGPUS=${NGPUS:-4}
NNODES=${NNODES:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Running experiment of method $TYPE"
torchrun --nproc_per_node $NGPUS --nnodes $NNODES \
        train_llama.py  \
        --ddp_timeout 18000 \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --resume_from_checkpoint false \
        --peft_type $TYPE \
        --bf16 true \
        --output_dir ./output/llama_$TYPE \
        --model_max_length 8192 \
        --use_flash_attn true \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 8     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 100     \
        --save_total_limit 1     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 50     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 10     \
        --deepspeed "config/ds_configs/stage2.json" \
        --tf32 true \
        --report_to none \
        --max_steps 1000
