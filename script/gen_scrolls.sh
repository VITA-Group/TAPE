#!/bin/sh
TYPE=${TYPE:-adape}
DATASET_NAME=${DATASET_NAME}
SAVE_DIR=${SAVE_DIR:-none}
torchrun --nproc_per_node=auto generate_scrolls.py --model_name output/scrolls/${DATASET_NAME}/${TYPE} --dataset_name ${DATASET_NAME} --save_dir ${SAVE_DIR}