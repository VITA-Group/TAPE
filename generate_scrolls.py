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
from finetune_scrolls import SUMMARY_TASKS, OTHER_TASKS
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

def generate_text_for_sum(input_ids, device="cpu"):
    beam_output=model.generate(torch.tensor(input_ids).long().unsqueeze(0).to(device),max_new_tokens=1000,num_return_sequences=1, top_k=1, do_sample=True)
    output=tokenizer.decode(beam_output[0][len(input_ids):],skip_special_tokens=True)
    return output

def generate_text_for_qa(input_ids, device="cpu"):
    beam_output=model.generate(torch.tensor(input_ids).long().unsqueeze(0).to(device),max_new_tokens=200,num_return_sequences=1, top_k=1, do_sample=True)
    output=tokenizer.decode(beam_output[0][len(input_ids):],skip_special_tokens=True)
    return output

generate_text = generate_text_for_sum if args.dataset_name in SUMMARY_TASKS else generate_text_for_qa
model_path = args.model_name
model_name = model_path.split('/')[-1]

sys.path.append(model_path)

# from model_llama_local import MyLlamaForCausalLM
from config_llama import MyLlamaConfig
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = MyLlamaConfig.from_pretrained(model_path)

# MODEL TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_path)
config.use_cache = True
torch_dtype = torch.float16
config.use_flash_attention_2 = 'flash'

module_name = config.rpe_type
MyLlamaForCausalLM = __import__(f"models.llama.{module_name}", fromlist=["MyLlamaForCausalLM"]).MyLlamaForCausalLM
model = MyLlamaForCausalLM.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True, torch_dtype=torch_dtype).to(device)

dataset = load_dataset(f"tau/scrolls", args.dataset_name)['test']
print("total examples: ", len(dataset))

genenerated_results = []
target_results = []


result_path = f"assets/results_scrolls/{args.dataset_name}"
create_recursive_directory(result_path)
for i in tqdm(range(len(dataset)), total=len(dataset)):
    example = dataset[i]
    if args.dataset_name in SUMMARY_TASKS:
        report = tokenizer("Context:\n" + example['input'] + "\n Please summarize this report:")
        report['input_ids'] = report['input_ids'][:7184] + report['input_ids'][-7:]
    else:
        report = tokenizer(" ".join(example['input'].split(" ")[:15000]))
        report['input_ids'] = report['input_ids'][:7991]

    generated = generate_text(report['input_ids'], device)

    genenerated_results.append({"prediction": generated, "id": example["id"]})
    # target_results.append(tokenizer.decode(summary['input_ids'][:1000], skip_special_tokens=True))

with open(f"{result_path}/{model_name}.json", 'w') as f:
    json.dump(genenerated_results, f)