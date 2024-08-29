#!/bin/sh
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

if [ "$DEBUG" = "true" ]; then
  step=1
  command="python -m pdb -m torch.distributed.launch"
else
  step=1
  command="torchrun"
fi

# module load anaconda3/2024.6
# conda activate mantis

echo "Running experiment of method $TYPE"
echo "The path to Python is: $(which python)"
echo "Command is $command"
$command --nproc_per_node $NGPUS --nnodes $NNODES \
        --rdzv_endpoint $head_node_ip:29500 \
        --rdzv_id $RANDOM \
        --rdzv_backend c10d \
        train_longlora.py  \
        --ddp_timeout 18000 \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --resume_from_checkpoint false \
        --peft_type $TYPE \
        --bf16 true \
        --output_dir ./output/llama_${TYPE} \
        --model_max_length 4096 \
        --use_flash_attn true \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 8     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 100     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 50     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps $step     \
        --deepspeed "config/ds_configs/stage2.json" \
        --tf32 true \
        --report_to none \
        --max_steps 1000
