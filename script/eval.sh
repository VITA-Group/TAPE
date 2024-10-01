name=${name:-adape_c4}
[ -z "${batch_size}" ] && batch_size=8
[ -z "${DATA_DIR}" ] && DATA_DIR=../data/pile # data path
# [ -z "${MODEL}" ] && MODEL=./output/adayarn_c4/ # checkpoint path
MODEL=./output/$name

for set in pg19 arxiv github
do
  for block_size in 1024 2048 3072 4096 5120 6144  # 
  do
    accelerate launch --main_process_port 25034 \
        --num_processes 2 \
        --config_file config/accelerate.yaml eval.py \
        --dataset_cache_dir ${DATA_DIR}_${set} \
        --block_size $block_size \
        --tokenizer_name ./models/llama/llama_tokenizer \
        --per_device_eval_batch_size $batch_size \
        --preprocessing_num_workers 48 \
        --model_name_or_path $MODEL
    done
done
