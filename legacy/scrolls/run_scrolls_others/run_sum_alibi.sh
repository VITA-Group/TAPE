# cd /mnt/bn/hzy-data-all/hierarchy_ape
# 'summ_screen_fd', 'qasper', 'qmsum', 'narrative_qa', 'gov_report', 'contract_nli', 'quality'
# "qasper" "narrative_qa" "contract_nli" "quality"
name=${name:-adape}
num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
for dataset_name in "qasper" "contract_nli" "quality"
do
batch_size=1
accelerate launch --mixed_precision fp16 --num_processes=$num_gpus --main_process_port 25401 train_scrolls_others.py \
    --dataset_name $dataset_name \
    --model_name_or_path ./output/${name} \
    --output_dir ./output/scrolls/${dataset_name}/${name} \
    --lr_scheduler_type polynomial \
    --config_name config/${name} \
    --preprocess alibi \
    --block_size 1024\
    --tokenizer_name ./models/llama/llama_tokenizer \
    --per_device_train_batch_size $batch_size\
    --per_device_eval_batch_size $batch_size\
    --gradient_accumulation_steps 8\
    --learning_rate 1e-5\
    --weight_decay 0.01\
    --max_train_steps 5000\
    --num_warmup_steps 50\
    --checkpointing_steps 50\
    --seed 2024 \
    --with_tracking \
    --preprocessing_num_workers 32 \
    --report_to tensorboard
done
