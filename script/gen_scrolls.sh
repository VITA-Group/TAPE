#!/bin/sh
TYPE=${TYPE:-adape}
dataset_name=${dataset_name}
srun torchrun --nproc_per_node=auto generate_scrolls_dist.py --model_name output/scrolls/${dataset_name}/${TYPE} --dataset_name ${dataset_name}