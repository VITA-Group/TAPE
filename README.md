<h1 align="center">
Adaptive Positional Encoding for Better Length Extrapolation (WIP)
</h1>

## Setup Environment
```shell
conda create -n adape python=3.10
conda activate adape
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
## Data
We use c4.en for pretraining and RedPajama-Data-1T-Sample for finetuning.

*Pretraining Data*
```shell
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "en/*"
```
or (not tested)
```python
from datasets import load_dataset
en = load_dataset("allenai/c4", "en")
```
*Finetuning Data*

This [line of code](https://github.com/zhuconv/AdaPE/blob/main/train_longlora.py#L214) will directly load data from hugginface hub or local cache dir.



## Pretraining
For pretraining, please download c4.en dataset

The scripts under script/ covers the commands for training and perpleixity evaluation.  For example, you can start training 151M BiPE-RoPE model with the following command:

```shell
OUTPUT_DIR=./output/adape  # path to save checkpoints and tensorboard
CONFIG_NAME=config/adarope.json
bash script/train.sh
```
You can change CONFIG_NAME to choose different positional encoding variants. (`choose from [config/adarope.json, config/alibi.json`)

## Evaluation
For pretraining perplexity evaluation, you need to prepare `monology/pile-test-val` using `download_data.py`. Then you can use the following command:
```shell
DATA_DIR=../data/pile  # path to load data
MODEL=./output/adape # model checkpoint path
bash script/eval.sh
```

For finetuning perplexity evaluation, you need to manually download data hosted by [LongLoRA](https://github.com/dvlab-research/LongLoRA/tree/main)

| Dataset    | Split      | Link                                                                                                         |
|:-----------|------------|--------------------------------------------------------------------------------------------------------------|
| PG19       | test       | [pg19/test.bin](https://drive.google.com/file/d/1QANDMdctpacPAYgS04adDXqByGEq-Ret/view?usp=share_link)       |
| Proof-pile | test       | [proof-pile/test_sampled_data.bin](https://drive.google.com/file/d/1bUI5lPDvrqzY_XXJJ2sSuvZx0Y9AZClE/view?usp=share_link)         |
 
 Then you can use the following command:
```shell
data=proof_pile  # path to load data
model_name=./output/llama_adape # model checkpoint path
bash script/long_eval.sh
```

## Credits
The codebase are inherited from [BiPE](https://github.com/zhenyuhe00/BiPE). Thanks to their excellent work!
