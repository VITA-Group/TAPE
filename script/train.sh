[ -z "${OUTPUT_DIR}" ] && OUTPUT_DIR=./output  # path to save checkpoints and tensorboard
[ -z "${DATA_DIR}" ] && DATA_DIR=./data  # path to load data
[ -z "${CONFIG_NAME}" ] && CONFIG_NAME=config/bipe_rope.json # choose from [config/bipe_rope.json, config/bipe_alibi.json, config/rope.json, config/alibi.json]


deepspeed --include localhost:0,1,2,3,4,5,6,7 train.py \
    --ddp_timeout 18000 \
    --dataset_cache_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --config_name $CONFIG_NAME \
    --max_steps 100000 \
    --warmup_steps 1000 \
    --lr_scheduler_type polynomial \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 50 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --model_max_position_embeddings 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --load_best_model_at_end True \
    --report_to "tensorboard" \
    --gradient_checkpointing False \
    --fp16 True
