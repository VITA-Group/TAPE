#!/bin/sh

# nodes=$( scontrol show hostnames $SLURM_JOB_NODELIST )
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# echo Node IP: $head_node_ip
# export LOGLEVEL=INFO

# if [[ "$SLURM_JOB_NAME" == "interactive" ]]; then
#   step=1
#   command="torchrun"
# else
#   step=1
#   command="srun torchrun"
# fi

# [ -z "${OUTPUT_DIR}" ] && OUTPUT_DIR=./output/${TYPE}_pile  # path to save checkpoints and tensorboard
# [ -z "${DATA_DIR}" ] && DATA_DIR=  # path to load data
# [ -z "${CONFIG_NAME}" ] && CONFIG_NAME=config/${TYPE}.json # choose from [config/bipe_rope.json, config/bipe_alibi.json, config/rope.json, config/alibi.json]

TYPE="yarn"

cd /home/ruisic/ssd/AdaPE/

torchrun --nproc_per_node=8 --nnodes=1 train.py \
        --use_flash_attention_2 flash \
        --ddp_timeout 18000 \
        --dataset_cache_dir ../data/c4 \
        --output_dir "output/${TYPE}_c4_new" \
        --config_name "config/${TYPE}.json" \
        --resume_from_checkpoint false \
        --max_steps 50000 \
        --warmup_steps 1000 \
        --lr_scheduler_type polynomial \
        --save_steps 1000 \
        --save_total_limit 1 \
        --eval_steps 1000 \
        --logging_steps 50 \
        --weight_decay 0.01 \
        --learning_rate 1e-4 \
        --model_max_position_embeddings 1024 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 \
        --do_train True \
        --do_eval True \
        --do_predict True \
        --evaluation_strategy "steps" \
        --save_strategy "steps" \
        --load_best_model_at_end True \
        --report_to "tensorboard" \
        --gradient_checkpointing False \
        --bf16 True
