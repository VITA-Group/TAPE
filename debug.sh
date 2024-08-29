export TYPE=adape
export NNODES=1
export NGPUS=4
export DEBUG=false
NTASKS=$((NNODES * NGPUS))
bash script/long_train.sh
# > log/train_llama_$TYPE.log 2>&1 &
    