#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import shutil
import time
from json import JSONDecodeError
from logging import getLogger
from pathlib import Path
from typing import Dict, List
import os
from datasets import load_dataset
from finetune_scrolls import SUMMARY_TASKS, OTHER_TASKS
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
import json

logger = getLogger(__name__)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)

def eval_data_dir(
    args,
    save_dir: str,
    bs: int = 8,
    local_rank=None,
) -> Dict:
    """Run evaluation on part of the data for one gpu and save to {save_dir}/rank_{rank}_output.json"""
    # model_name = str(model_name)
    assert local_rank is not None
    torch.distributed.init_process_group(backend="nccl", rank=local_rank)
    print("local_rank", local_rank)

    save_dir = Path(save_dir)
    save_path = save_dir.joinpath(f"rank_{local_rank}_output.json")
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'

    def generate_text_for_sum(input_ids, device="cpu"):
        beam_output=model.generate(torch.tensor(input_ids).long().unsqueeze(0).to(device),max_new_tokens=1000, num_return_sequences=1, top_k=1, do_sample=False, use_cache=config.use_cache)
        output=tokenizer.decode(beam_output[0][len(input_ids):], skip_special_tokens=True)
        return output

    def generate_text_for_qa(input_ids, device="cpu"):
        beam_output=model.generate(torch.tensor(input_ids).long().unsqueeze(0).to(device),max_new_tokens=200,num_return_sequences=1, top_k=1, do_sample=True, use_cache=config.use_cache)
        output=tokenizer.decode(beam_output[0][len(input_ids):],skip_special_tokens=True)
        return output

    generate_text = generate_text_for_sum if args.dataset_name in SUMMARY_TASKS else generate_text_for_qa
    model_path = args.model_name
    

    # from model_llama_local import MyLlamaForCausalLM
    from config_llama import MyLlamaConfig
    config = MyLlamaConfig.from_pretrained(model_path)

    # MODEL TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if 'adape' in model_path:
        print("use_cache disabled for adape")
        config.use_cache = False
    else:
        print(f"use_cache enabled")
        config.use_cache = True

    torch_dtype = torch.float16
    config.use_flash_attention_2 = 'flash'

    module_name = config.rpe_type
    context_size = 8192
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    if module_name == "adape":
        print("Use TAPE now!")
        from models.llama.adape import MyLlamaForCausalLM
    # MyLlamaForCausalLM = __import__(f"models.llama.{module_name}", fromlist=["MyLlamaForCausalLM"]).MyLlamaForCausalLM
    else:
        from transformers import AutoModelForCausalLM as MyLlamaForCausalLM
    model = MyLlamaForCausalLM.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True)

    dataset = load_dataset(f"tau/scrolls", args.dataset_name)['validation'].select_columns(['id', 'input'])
    dataset = dataset.map(lambda e: {'id': e['id'], 'input': e['input']})
    print("total examples: ", len(dataset))
    # I set shuffle=True for a more accurate progress bar.
    # If all the longest samples are first, the prog bar estimate is too high at the beginning.
    # print(dataset)
    sampler = DistributedSampler(dataset, shuffle=False)
    # sampler = ds.make_sortish_sampler(bs, distributed=True, add_extra_examples=False, shuffle=True)
    data_loader = DataLoader(dataset, sampler=sampler, )
    results = {}
    for example in tqdm(data_loader):
        # example = batch[0]
        assert len(example["id"]) == 1
        if args.dataset_name in SUMMARY_TASKS:
            report = tokenizer("Context:\n" + example['input'][0] + "\n Please summarize this report:")
            report['input_ids'] = report['input_ids'][:7184] + report['input_ids'][-7:]
        else:
            report = tokenizer(" ".join(example['input'][0].split(" ")[:15000]))
            report['input_ids'] = report['input_ids'][:7991]

        generated = generate_text(report['input_ids'], device=model.device)
        results[example["id"][0]] = generated
        # results.append({"prediction": generated, "id": example["id"]})
        # for i, pred in enumerate(preds):
        #     results.append({"pred": pred, "id": ids[i].item()})
    save_json(results, save_path)
    return results, sampler.num_replicas


def run_generate():
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

    parser.add_argument("--save_dir", type=str, help="where to save", default=None)
    parser.add_argument("--max_source_length", type=int, default=None)
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument(
        "--local_rank", type=int, default=-1, required=False, help="should be passed by distributed.launch"
    )
    # parser.add_argument(
    #     "--num_return_sequences", type=int, default=1, required=False, help="How many sequences to return"
    # )
    parser.add_argument(
        "--sync_timeout",
        type=int,
        default=600,
        required=False,
        help="How long should master process wait for other processes to finish.",
    )
    # start_time = time.time()
    # args, rest = parser.parse_known_args()
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f"assets/results_scrolls/{args.dataset_name}"
    json_save_dir = Path(args.save_dir + "_tmp")
    Path(json_save_dir).mkdir(exist_ok=True)  # this handles locking.
    intermediate_files = list(json_save_dir.glob("rank_*.json"))
    if intermediate_files:
        raise ValueError(f"Found files at {json_save_dir} please move or remove them.")
        # In theory, a node could finish and save before another node hits this. If this happens, we can address later.


    Path(args.save_dir).mkdir(exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    results, num_replicas = eval_data_dir(
        args,
        json_save_dir,
        local_rank=local_rank,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    if local_rank <= 0:
        partial_results = gather_results_from_each_node(num_replicas, json_save_dir, args.sync_timeout)
        final_results = combine_partial_results(partial_results)
        save_path = save_dir.joinpath(f"{args.model_name.split('/')[-1]}.json")
        print(f"Saving aggregated results at {save_path}, intermediate in {json_save_dir}/")
        save_json(final_results, save_path)
        shutil.rmtree(json_save_dir)


def combine_partial_results(partial_results) -> List:
    """Concatenate partial results into one file, then sort it by id."""
    merged_dict = {}
    for d in partial_results:
        merged_dict.update(d)
    return merged_dict


def lmap(f , x) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def gather_results_from_each_node(num_replicas, save_dir, timeout) -> List[Dict[str, List]]:
    # WAIT FOR lots of .json files
    start_wait = time.time()
    logger.info("waiting for all nodes to finish")
    json_data = None
    while (time.time() - start_wait) < timeout:
        json_files = list(save_dir.glob("rank_*.json"))
        if len(json_files) < num_replicas:
            continue
        try:
            # make sure all json files are fully saved
            json_data = lmap(load_json, json_files)
            return json_data
        except JSONDecodeError:
            continue
    else:
        raise TimeoutError("Rank 0 gave up on waiting for other processes")
    # Unreachable


if __name__ == "__main__":
    # Usage for MT:
    run_generate()