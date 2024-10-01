export TYPE=${TYPE:-adape}
export NNODES=${NNODES:-1}
export NGPUS=${NGPUS:-4}
# NTASKS=$((NNODES * NGPUS))
if [[ "$SLURM_JOB_NAME" == "interactive" ]]; then
  NNODES=1 TYPE=rope bash script/sptest.sh # > log/train_llama_$TYPE.log 2>&1 &
else
  sbatch -o log/test_speed.log \
        -t 12:00:00 \
        -J "test_speed" \
        --ntasks=$NNODES \
        --cpus-per-task=48 \
        --gpus-per-task=$NGPUS \
        --nodes=$NNODES \
        --gres=gpu:$NGPUS \
        --constraint="gpu80&pcie" \
        --mail-type=begin,end \
        --mail-user=zhuconv@gmail.com \
        script/sptest.sh
fi
# sbatch -t 24:00:00 -o log/_neweval_mantis.log -J "mm_eval" --ntasks=1 --nodes=1 --cpus-per-task=12 --mem=100G --gres=gpu:1 --constraint="gpu80" --mail-type=all --mail-user=zhuconv@gmail.com script/eval_mllava.sh