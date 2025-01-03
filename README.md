<h1 align="center">Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding (TAPE)</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2501.00712"><img src="https://img.shields.io/badge/arXiv-2501.00712-B31B1B?labelColor=white&logo=data:image/vnd.microsoft.icon;base64,AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACUgswAlILMAJSCzACUgswAlILMAJSCzAKWsswClrLMApayzAKWsswelrLOPpayzvKWsswulrLMAAAAAAAAAAAAlILMAJSCzMyUgs2klILMNJSCzACUgswClrLMApayzAKWsswmlrLOFpayz7aWss2WlrLMApayzAAAAAAAAAAAAJSCzACUgs1glILP3JSCzmCUgsxAiHbMAqbCzAKWsswylrLOOpayz+aWss4OlrLMDpayzAKWsswAAAAAAAAAAACUgswAlILMJJSCzliUgs/8lILOkGBKzFbG5sw6lrLOWpayz/6Wss6KlrLMMpayzAKWsswAAAAAAAAAAAAAAAAAlILMAJSCzACUgsxUlILO3JB+z/zIus7KfpbOmpayz/6Wss7+lrLMapayzAKWsswClrLMAAAAAAAAAAAAAAAAAJSCzACUgswAjHbMAHRezKDMvs9J3erP/payz/6Srs9ipsLMtpq2zAKOqswClrLMAAAAAAAAAAAAAAAAAAAAAAAAAAACepLMApayzAKeusxuZn7PApayz/5mfs/9RULPRHBazLCciswAAALMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAApayzAKWsswGlrLOUpq2z/6Cns/9YV7P/JB+z/yUgs7ElILMIJSCzAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKWsswCaoLMApq2ziKSrs/9maLP/JyKz/yQfs/8lILOiJSCzBiUgswAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACChbMAlZuzALG5sxRlZrO8LCiz/yUgs/8xLbPDJB+zHScjswArJ7MAAAAAAAAAAAAAAAAAAAAAAAAAAAAlILMAJSCzACMeswAjHrNAIx6z4iQfs/9QT7P/lpyz16y0sy+nrrMApayzAKWsswAAAAAAAAAAAAAAAAAlILMAJSCzACUgswAlILMoJSCz0CUgs/8lILOkkZazg6ats/elrLPEpayzHqWsswClrLMApayzAAAAAAAAAAAAJSCzACUgswAlILMVJSCztiUgs/8lILOdIRuzEf//swGlrLNppayz8KWss6ulrLMRpayzAKWsswAAAAAAJSCzACUgswAlILMIJSCzlyUgs/4lILOWJSCzDyMeswCvt7MApayzAKWss1mlrLPkpayzbqWsswClrLMAAAAAACUgswAlILMBJSCzdSUgs/UlILOOJSCzDCUgswAlILMApayzAKWsswClrLMApayzN6WssyqlrLMApayzAAAAAAAlILMAJSCzDCUgs74lILObJSCzCyUgswAlILMAJSCzAAAAAAClrLMApayzAKWsswClrLMApayzAKWsswAAAAAAwAAAAMAAAADAAAAAwAEAAMABAADAAwAA4AcAAOAHAADgBwAA4AcAAMADAACAAQAAgAEAAAABAAAAAQAAAIEAAA==" alt="arXiv"></a>
    <a href="https://openreview.net/forum?id=Us1RXG1Ji2&noteId=Us1RXG1Ji2"> <img alt="OpenReview" src="https://img.shields.io/badge/ICLR%2725-OpenReview-blue?logoSize=auto&labelColor=white&logo=data:image/vnd.microsoft.icon;base64,AAABAAEAEBAAAAAAIACOAgAAFgAAAIlQTkcNChoKAAAADUlIRFIAAAAQAAAAEAgGAAAAH/P/YQAAAARnQU1BAACxjwv8YQUAAAABc1JHQgCuzhzpAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAIMSURBVDiNtVJfSFNRGP+dsytmrfBBH9zyLrkyqdFTlijO7h5WV/AhCDF33ZKiXqKHeo0IovYQFPRQEET+Y6uek1q9LDfJbc2XRoWylX/mejDMaaS2vF8Pecc21IeiHxz4vnPO9/t+v/Md4B/BNtt0KCoV5hrR1ZGX/hvbMsVm9l4AgNiseT0+V1NVeCY7XWdLSXVwPTgipu/F0qaWnxq3HjJ9WQikxL6nExIBADOwhwAgH3cltiSIz9XYofEwZ0i++GRZ+pYr6+1qSDHv+IHLwYBPt5osVSIAQDRtGmw0ZzxEf3L/hLSczQkAgOy64TaAOyuLazsrKst/FL4REYbzTKGp2l8A8Oh9nR0ArsdtHVcSklwq2aGopK+8hZEpkdr2zQqDH+vpjO1zeKPzs5sHU69Pj7aTe+x8q06ggR7osdPZs0sITZtvtVlmWN8HacizP8muvbV5V3esvspkDYfdoY6BgdZhBgDu6GPrUFP3JCOc04ef41ps038AAJ7wibsaE09q3NLOmPjO19zJHIpKwYCP6fKnJ8cqhdLC3nDj93573Ei8rgxMNPtbLiUAsFPReXrSVF3U0GJtXswTXHxeX760u8rbb48YAUBjUgOx2rX8bc6OAYCsuMa1Fa2aV/D5os6dwaNGz2g3AUDPm/tFs+6KLLgBQI183eNQ1OUNC1va3xay4sr8VeF/w28RA8Eb9erO/QAAAABJRU5ErkJggg==" alt="OpenReview"> </a>
    <a href="https://github.com/VITA-Group/TAPE"><img src="https://img.shields.io/github/stars/VITA-Group/TAPE"></a>
</p>


This repository contains the official implementation of TAPE as described in the paper: [Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding](https://arxiv.org/abs/2501.00712) by Jiajun Zhu, Peihao Wang, Ruisi Cai, Jason D. Lee, Pan Li, Zhangyang Wang.


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
