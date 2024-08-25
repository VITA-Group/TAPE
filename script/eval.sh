[ -z "${batch_size}" ] && batch_size=8
[ -z "${DATA_DIR}" ] && DATA_DIR=./data/pile # data path
[ -z "${MODEL}" ] && MODEL=./output/rope_pile/ # checkpoint path

for block_size in 1024 2048 3072 4096 5120 6144
do
    for set in pg19 arxiv github
    do
        accelerate launch --main_process_port 25034 --config_file config/accelerate.yaml eval.py \
            --dataset_cache_dir ${DATA_DIR}_${set} \
            --block_size $block_size \
            --tokenizer_name ./models/llama/llama_tokenizer \
            --per_device_eval_batch_size $batch_size \
            --preprocessing_num_workers 64 \
            --model_name_or_path $MODEL
    done
done

