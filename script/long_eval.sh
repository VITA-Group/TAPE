torchrun --master_port 29512 --nproc_per_node=auto eval_longlora.py --seq_len 8192 --context_size 8192 --batch_size 1 --base_model /zhujiajun/model/Llama-2-7b-hf --peft_model ./output/llama_lora/checkpoint-1000 --data_path /zhujiajun/data/pg19/test.bin > log/eval_longlora.log 2>&1 &
# python3 eval_retrivial.py \
#         --context_size 8096 \
#         --base_model path_to/Llama-2-7b-longlora-32k \
#         --max_tokens 8096 \
#         --interval 1000

