[ -z "${OUTPUT_DIR}" ] && OUTPUT_DIR=./output/finetune/big_rope_pile  # path to save checkpoints and tensorboard
[ -z "${DATA_DIR}" ] && DATA_DIR=/zhujiajun/data/pile  # path to load data
[ -z "${CONFIG_NAME}" ] && CONFIG_NAME=config/new_rope.json # choose from [config/bipe_rope.json, config/bipe_alibi.json, config/rope.json, config/alibi.json]


deepspeed --master_port 25012 --include localhost:0,1,2,3,4,5,6,7 train.py \
    --deepspeed ./ds_config.json \
    --ddp_timeout 18000 \
    --dataset_cache_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --config_name $CONFIG_NAME \
    --model_name_or_path ./output/train/rope_pile/checkpoint-20000 \
    --resume_from_checkpoint false \
    --max_steps 10000 \
    --warmup_steps 1000 \
    --lr_scheduler_type polynomial \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 50 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --model_max_position_embeddings 1024 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --load_best_model_at_end True \
    --report_to "tensorboard" \
    --gradient_checkpointing False \
    > finetune_big_rope.log 2>&1 &