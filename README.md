<h1 align="center">Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding (TAPE)</h1>
<p align="center">
    <a href=""><img src="https://img.shields.io/badge/-arXiv-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
    <a href=""><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href=""> <img alt="License" src="https://img.shields.io/static/v1?label=UR&message=ICLR%2725&color=blue"> </a>
</p>


This repository contains the official implementation of TAPE as described in the paper: [Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding]() by Jiajun Zhu, Peihao Wang, Ruisi Cai, Jason D. Lee, Pan Li, Zhangyang Wang.


## Getting Started
```shell
conda create -n adape python=3.10
conda activate adape
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Arithmetic Learning
It is implemented independently under the directory `arithmetic/`, another github repo linked as submodule. It should has independent environment as well. Please refer to its [README](https://github.com/zhuconv/arithmetic/blob/main/README.md) for detailed instructions about training and evaluation.

## Training From Scratch
### Pretraining
The scripts under script/ covers the commands for training. For example, you can start training TAPE (`adape` in code) model with the following command:

```shell
export TYPE=adape
bash script/train.sh
```
You can change CONFIG_NAME to choose different positional encoding variants. (`choose from those under config/`)

### Finetuning on SCROLLS and Evaluation
There are three steps to get evaluation results:
1. finetune pre-trained models on SCROLLS
2. generate answers in validation set
3. evaluate the answers with corresponding metric

```shell
export TYPE=adape DATASET_NAME=quality
export METRIC_DIR=scrolls/metrics
export SAVE_DIR=scrolls/quality
bash script/ft_scrolls.sh # assume the pretrained checkpoint is under output/${TYPE}_c4, if not, need to set 'output_name=<your_output_name>'
bash script/gen_scrolls.sh
python eval_scrolls.py --split validation --dataset_name $DATASET_NAME --predictions ${SAVE_DIR}/${TYPE}.json  --metrics_output_dir $METRIC_DIR
```

You can change DATASET_NAME to choose different dataset. (`choose from ['narrative_qa', 'quality', "qasper", 'contract_nli']`)

## PEFT Llama2-7B
### Finetuning
Similiar to training from scratch, you can use the following command ans select different methods: 
```shell
export TYPE=adape
bash script/train_llama.sh
```

### Evaluation
For finetuning perplexity evaluation, you need to manually download data hosted by [LongLoRA](https://github.com/dvlab-research/LongLoRA/tree/main)

| Dataset    | Split      | Link                                                                                                         |
|:-----------|------------|--------------------------------------------------------------------------------------------------------------|
| PG19       | test       | [pg19/test.bin](https://drive.google.com/file/d/1QANDMdctpacPAYgS04adDXqByGEq-Ret/view?usp=share_link)       |
| Proof-pile | test       | [proof-pile/test_sampled_data.bin](https://drive.google.com/file/d/1bUI5lPDvrqzY_XXJJ2sSuvZx0Y9AZClE/view?usp=share_link)         |
 
 Then you can use the following command:
```shell
data=proof_pile
model_path=output/llama_adape
bash script/eval_llama.sh
```

We also have `eval_retrieval.py` for evaluation on passkey retrieval task.
```shell
python3 eval_retrieval.py --context_size 8192 --base_model output/llama_adape --max_tokens 8192 --interval 1000
```


## Credits
The codebase are inherited from [BiPE](https://github.com/zhenyuhe00/BiPE) and [LongLoRA](https://github.com/dvlab-research/LongLoRA/tree/main). Thanks to their excellent work!
