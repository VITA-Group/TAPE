cd /mnt/bn/hzy-data-all/hierarchy_ape
# "qasper" "contract_nli" "quality" "narrative_qa"
for dataset_name in "narrative_qa"
do
batch_size=1
accelerate launch --mixed_precision fp16 --num_processes=8 --main_process_port 25401 gov_report/train_scrolls_others.py\
    --dataset_name $dataset_name \
    --model_name_or_path /mnt/bn/hzy-data-all/output_pile/ape1_sent_alibi_div96/step_500000 \
    --dataset_cache_dir /mnt/bn/hzy-data-all/hierarchy_ape/gov_report/scrolls_cache/${dataset_name}\
    --output_dir /mnt/bn/hzy-data-all/output_finetune_scrolls/${dataset_name}/ape1_sent_alibi_div96 \
    --lr_scheduler_type polynomial \
    --preprocess ape1_sent_alibi_div96 \
    --block_size 1024\
    --tokenizer_name llama_tokenizer\
    --config_name config/ape1_sent_alibi_div96\
    --per_device_train_batch_size $batch_size\
    --per_device_eval_batch_size $batch_size\
    --gradient_accumulation_steps 8\
    --learning_rate 1e-5\
    --weight_decay 0.01\
    --max_train_steps 5000\
    --num_warmup_steps 50\
    --checkpointing_steps 50\
    --seed 2023\
    --with_tracking\
    --preprocessing_num_workers 32\
    --report_to tensorboard
done



cd /mnt/bn/hzy-data-all/hierarchy_ape
# "gov_report" "summ_screen_fd" "qmsum"
for dataset_name in "gov_report" "summ_screen_fd" "qmsum"
do
batch_size=1
accelerate launch --mixed_precision fp16 --num_processes=8 --main_process_port 25401 gov_report/train_scrolls.py\
    --dataset_name $dataset_name \
    --model_name_or_path /mnt/bn/hzy-data-all/output_pile/ape1_sent_alibi_div96/step_500000 \
    --dataset_cache_dir /mnt/bn/hzy-data-all/hierarchy_ape/gov_report/scrolls_cache/${dataset_name}\
    --output_dir /mnt/bn/hzy-data-all/output_finetune_scrolls/${dataset_name}/ape1_sent_alibi_div96 \
    --lr_scheduler_type polynomial \
    --preprocess ape1_sent_alibi_div96 \
    --block_size 1024\
    --tokenizer_name llama_tokenizer\
    --config_name config/ape1_sent_alibi_div96\
    --per_device_train_batch_size $batch_size\
    --per_device_eval_batch_size $batch_size\
    --gradient_accumulation_steps 8\
    --learning_rate 1e-5\
    --weight_decay 0.01\
    --max_train_steps 5000\
    --num_warmup_steps 50\
    --checkpointing_steps 50\
    --seed 2023\
    --with_tracking\
    --preprocessing_num_workers 8\
    --report_to tensorboard
done