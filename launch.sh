export TYPE=adape
export NNODES=1
export NGPUS=4
NTASKS=$((NNODES * NGPUS))
if [ -z "$SLURM_JOB_NAME" ] || [[ "$SLURM_JOB_NAME" == "interactive" ]]; then
  bash script/long_train.sh # > log/train_llama_$TYPE.log 2>&1 &
else
  sbatch -o log/train_llama_$TYPE.log \
        -t 1:00:00 \
        -J $TYPE \
        --ntasks=$NTASKS \
        --cpus-per-task=12 \
        --constraint=gpu80 \
        --nodes=$NNODES \
        --gres=gpu:$NGPUS \
        --mail-type=begin \
        --mail-type=end \
        --mail-user=zhuconv@gmail.com \
        script/long_train.sh
fi