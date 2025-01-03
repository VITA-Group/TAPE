#!/bin/sh
nodes=$( scontrol show hostnames $SLURM_JOB_NODELIST )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

NGPUS=${NGPUS:-4}
NNODES=${NNODES:-1}
max_steps=$((10000 / NNODES))
# Check TYPE and modify TAPE accordingly
# if [[ $TYPE == "rope" || $TYPE == "xpos" ]]; then
#     TAPE="larg/$TAPE"
# fi
torchrun --nproc_per_node $NGPUS --nnodes $NNODES \
        --rdzv_endpoint $head_node_ip:29523 \
        --rdzv_id $RANDOM \
        --rdzv_backend c10d \
        train.py \
        --use_flash_attention_2 flash \
        --deepspeed "config/ds_configs/stage2.json" \
        --ddp_timeout 18000 \
        --dataset_cache_dir ../data/c4 \
        --output_dir "output/${TYPE}_c4" \
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
