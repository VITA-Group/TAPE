export TYPE=adape
export NNODES=1
export NGPUS=4
# NTASKS=$((NNODES * NGPUS))
if [[ "$SLURM_JOB_NAME" == "interactive" ]]; then
  bash script/train_mllava.sh # > log/train_llama_$TYPE.log 2>&1 &
else
  sbatch -o log/train_mm_$TYPE.log \
        -t 24:00:00 \
        -J "mm_$TYPE" \
        --ntasks=$NNODES \
        --cpus-per-task=48 \
        --gpus-per-task=$NGPUS \
        --nodes=$NNODES \
        --gres=gpu:$NGPUS \
        --mail-type=all \
        --mail-user=zhuconv@gmail.com \
        script/train_mllava.sh
fi
