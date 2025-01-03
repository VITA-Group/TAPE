#!/bin/sh
TYPE=${TYPE:-adape}
DATASET_NAME=${DATASET_NAME:-quality} # 'narrative_qa', 'quality', "qasper", 'contract_nli' 
use_llama=${use_llama:-false}
if [[ "$use_llama" == "false" ]]; then
MODEL_NAME=output/scrolls/${DATASET_NAME}/${TYPE}
SAVE_DIR=${SAVE_DIR:-"assets/results_scrolls/summarization/${DATASET_NAME}"}
SCRIPT=generate_scrolls.py
else
MODEL_NAME=output/llama_${TYPE} # thetalora, longlora, lora, adape
SAVE_DIR=${SAVE_DIR:-"assets/llama_scrolls/${DATASET_NAME}"}
SCRIPT=generate_llama_scrolls.py
fi
torchrun --nproc_per_node=2 $SCRIPT --model_name ${MODEL_NAME} --dataset_name ${DATASET_NAME} --save_dir ${SAVE_DIR}