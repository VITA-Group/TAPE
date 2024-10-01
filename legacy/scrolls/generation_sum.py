from transformers import pipeline
from datasets import load_dataset, load_from_disk
import torch
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import json
from tqdm import tqdm
import sys
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="ape1_sent_rope",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="contract_nli",
)
parser.add_argument(
    "--step",
    type=int,
    default="500",
)


args = parser.parse_args()
print(args)

def create_recursive_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_text(input_ids, device="cpu"):
    beam_output=model.generate(torch.tensor(input_ids).long().unsqueeze(0).to(device),max_new_tokens=1000,num_return_sequences=1, top_k=1)
    output=tokenizer.decode(beam_output[0][len(input_ids):],skip_special_tokens=True)
    return output

model_name = args.model_name
model_path = f'/mnt/bn/hzy-data-all/output_finetune_scrolls/{args.dataset_name}/{model_name}/step_{args.step}'
sys.path.append(model_path)
from model_llama_local import MyLlamaForCausalLM
from config_llama_local import MyLlamaConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = MyLlamaConfig.from_pretrained(model_path)
if "ape1" in model_name or "sin1" in model_name:
    config.use_cache = False
else:
    config.use_cache=True
if model_name == "sin1":
    config.use_cache=True
    
model = MyLlamaForCausalLM.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True).to(device)
if "sin1" in model_name:
    model.model.init_sin_embeddings_new()
    model.model.pe = model.model.pe.to(device)

dataset = load_from_disk(f"/mnt/bn/hzy-data-all/hierarchy_ape/gov_report/scrolls/{args.dataset_name}")['test']
print("total examples: ", len(dataset))

genenerated_results = []
target_results = []


create_recursive_directory(f"/mnt/bn/hzy-data-all/hierarchy_ape/gov_report/results_scrolls/{args.dataset_name}/{model_name}")
for i in tqdm(range(len(dataset)), total=len(dataset)):
    example = dataset[i]
    report = tokenizer("Context:\n" + example['input'] + "\n Please summarize this report:")

    # summary = tokenizer(example['output'])

    report['input_ids'] = report['input_ids'][:7184] + report['input_ids'][-7:]

    generated = generate_text(report['input_ids'], device)

    genenerated_results.append({"prediction": generated, "id": example["id"]})
    # target_results.append(tokenizer.decode(summary['input_ids'][:1000], skip_special_tokens=True))

with open(f"/mnt/bn/hzy-data-all/hierarchy_ape/gov_report/results_scrolls/{args.dataset_name}/{model_name}/{model_name}_generated.json", 'w') as f:
    json.dump(genenerated_results, f)
# with open(f"/mnt/bn/hzy-data-all/hierarchy_ape/gov_report/results_scrolls/{args.dataset_name}/{model_name}/{model_name}_target.json", 'w') as f:
#     json.dump(target_results, f)