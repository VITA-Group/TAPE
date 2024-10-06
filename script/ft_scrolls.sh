#!/bin/sh
TYPE=${TYPE:-adape}
output_name=${output_name:-$TYPE}
echo output_name="$output_name"
dataset_name=${dataset_name:-quality}
echo "dataset_name=${dataset_name}"
# summary include "gov_report" "summ_screen_fd" "qmsum"
# others include 'narrative_qa', 'quality', "qasper", 'contract_nli'
CUDA_ALLOC_CONF=expandable_segments:True
# TORCH_DISTRIBUTED_DEBUG=DETAIL
batch_size=2
# nproc_per_node should be 4 in formal training
torchrun --nproc_per_node 4 --nnodes 1 \
    finetune_scrolls.py \
    --bf16 True \
    --deepspeed "config/ds_configs/stage2.json" \
    --dataset_name $dataset_name \
    --model_name_or_path output/${output_name}_c4 \
    --output_dir output/scrolls/${dataset_name}/${output_name} \
    --lr_scheduler_type polynomial \
    --block_size 1024 \
    --use_flash_attention_2 flash \
    --tokenizer_name models/llama/llama_tokenizer \
    --config_name config/${TYPE}.json \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --max_steps 1000 \
    --warmup_ratio 0.1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --seed 2024 \
    --preprocessing_num_workers 32 \
    --resume_from_checkpoint True \
    --report_to tensorboard
