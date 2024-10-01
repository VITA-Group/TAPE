#!/bin/sh
nodes=$( scontrol show hostnames $SLURM_JOB_NODELIST )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

if [[ "$SLURM_JOB_NAME" == "interactive" ]]; then
  step=1
  command="torchrun"
else
  step=1
  command="srun torchrun"
fi

# [ -z "${OUTPUT_DIR}" ] && OUTPUT_DIR=./output/${TYPE}_pile  # path to save checkpoints and tensorboard
# [ -z "${DATA_DIR}" ] && DATA_DIR=  # path to load data
# [ -z "${CONFIG_NAME}" ] && CONFIG_NAME=config/${TYPE}.json # choose from [config/bipe_rope.json, config/bipe_alibi.json, config/rope.json, config/alibi.json]

for use_flash_attention_2 in none flash triton 3triton; do
  for TYPE in adape; do
    $command --nproc_per_node $NGPUS --nnodes $NNODES \
            --rdzv_endpoint $head_node_ip:29512 \
            --rdzv_id $RANDOM \
            --rdzv_backend c10d \
            train.py \
            --use_flash_attention_2 ${use_flash_attention_2} \
            --deepspeed "config/ds_configs/stage2.json" \
            --ddp_timeout 18000 \
            --dataset_cache_dir ../data/c4 \
            --output_dir "output/sptest/${TYPE}_${use_flash_attention_2}" \
            --config_name "config/${TYPE}.json" \
            --resume_from_checkpoint false \
            --max_steps 100 \
            --warmup_steps 10 \
            --lr_scheduler_type polynomial \
            --save_steps 10 \
            --save_total_limit 1 \
            --eval_steps 10 \
            --logging_steps 10 \
            --weight_decay 0.01 \
            --learning_rate 1e-4 \
            --model_max_position_embeddings 1024 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 4 \
            --do_train True \
            --do_eval True \
            --do_predict True \
            --evaluation_strategy "steps" \
            --save_strategy "steps" \
            --load_best_model_at_end True \
            --report_to "tensorboard" \
            --gradient_checkpointing False \
            --bf16 True
  done
done
