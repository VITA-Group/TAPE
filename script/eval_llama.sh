#!/bin/bash
model_name=${model_name:-llama_adape}
model_path=${model_path:-output/${model_name}}
seq_len=${seq_len:-8192}
data=${data:-proof_pile} # pg19_test

torchrun --master_port 29523 --nproc_per_node=auto eval_llama.py --seq_len $seq_len --context_size 8192 --batch_size 1 --base_model $model_path --data_path ../data/${data}.bin > log/eval_tune/${data}_${model_name}_${seq_len}.log 2>&1 &


