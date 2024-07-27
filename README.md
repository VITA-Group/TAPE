<h1 align="center">
Adaptive Positional Encoding for Better Length Extrapolation (WIP)
</h1>

## Setup Environment
```shell
conda create -n bipe python=3.10
conda activate bipe
pip3 install -r requirements.txt
```

## Data for Pretraining
We use [the Pile](uncopyrighted) for pretraining with all copyrighted data removed.
```shell
cd BiPE;
DATA_DIR=./data # the directory to save the data
python3 download_data.py --dataset-cache-dir $DATA_DIR
```

## Pretraining
The scripts under script/ covers the commands for training and perpleixity evaluation.   

For training, the key modifications for BiPE are getting token ids (intra-segment) and position ids (inter-segment) by the `get_bilevel_ids` function. Then, the token ids are used to get absolute positional encodings (`get_ape_embeddings`) and the position ids are used to get relative positional encodings. For example, you can start training 151M BiPE-RoPE model with the following command:
```shell
cd BiPE
OUTPUT_DIR=./output  # path to save checkpoints and tensorboard
DATA_DIR=./data  # path to load data
CONFIG_NAME=config/bipe_rope.json
bash script/train.sh
```
You can change CONFIG_NAME to choose different positional encoding variants. (`choose from [config/bipe_rope.json, config/bipe_alibi.json, config/rope.json, config/alibi.json`)

## Perplexity Evaluation
For perplexity evaluation, you can use the following command:
```shell
cd BiPE;
DATA_DIR=./data  # path to load data
MODEL=./bipe_rope # model checkpoint path
bash script/eval.sh
```

## Credits
The codebase are inherited from [BiPE](https://github.com/zhenyuhe00/BiPE). Thanks to their excellent work!
