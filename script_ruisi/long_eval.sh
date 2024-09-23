#!/bin/bash
model_name=${model_name:-llama_adape}
model_path="./output/${model_name}"
seq_len=${seq_len:-8192}
data=${data:-proof_pile} # pg19_test
# model_name="llama_base"
# model_path="/scratch/gpfs/pw4811/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
torchrun --master_port 29523 --nproc_per_node=auto eval_longlora.py --seq_len $seq_len --context_size 8192 --batch_size 1 --base_model $model_path --data_path ../data/${data}.bin > log/eval_tune/${data}_${model_name}_${seq_len}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python3 eval_retrieval.py \
#         --context_size 32768 \
#         --base_model $model_path \
#         --max_tokens 32768 \
#         --interval 1000 > log/retrieval_$model_name.log 2>&1 &

