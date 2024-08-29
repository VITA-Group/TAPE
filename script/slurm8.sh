#!/bin/bash
export SBATCH_JOB_NAME="adape"
export SBATCH_NODES=2
export SBATCH_NTASKS=8
export SBATCH_CPUS_PER_TASK=12
export SBATCH_MEM_PER_CPU="4G"
export SBATCH_GRES="gpu:4"
export SBATCH_TIME="00:24:00"
export SBATCH_CONSTRAINT="gpu80"