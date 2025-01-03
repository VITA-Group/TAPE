#    Modification Copyright 2024 Jiajun Zhu
#    Modification Copyright 2024 Zhenyu He
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import os
from itertools import chain
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed
import transformers
from config_llama import MyLlamaConfig
from transformers import Trainer, default_data_collator, AutoTokenizer
from datasets import load_dataset, IterableDataset

transformers.logging.set_verbosity_info()

@dataclass
class ModelArguments:
    config_name: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    dataset_cache_dir: str = field(default=None, metadata={"help": "Path to the data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    use_flash_attention_2: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_position_embeddings: int = field(
        default=1024,
        metadata={"help": "Maximum position embeddings."},
    )
    rope_scaling_type: Optional[str] = field(default=None)
    rope_scaling_factor: float = field(default=1.0)
    resume_from_checkpoint: Optional[bool] = field(default=None)
    finetune_from_pretrained: Optional[str] = field(default=None)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def load_json_dataset(training_args, dataset_dir, sanity_check=False, streaming=True):
    import os, glob, random, copy
    dataset_subsample_rate = 0.1
    test_split_percentage = 0.03
    def uniform_sample_list(file_list, subsample_rate):
        if not 0 < subsample_rate <= 1:
            raise ValueError(f'subsample_rate wrong: {subsample_rate}')

        sample_size = int(len(file_list) * subsample_rate)
        return random.sample(file_list, sample_size)
    def print_rank_0(*msg):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if local_rank != 0:
            return
        print(*msg)

    if not os.path.exists(dataset_dir):
        raise ValueError(f'The sepcified data path does not exist: {dataset_dir}')

    data_files = {
        'train': [],
        'validation': [],
        'test': []
    }

    # json_suffices = ['jsonl.zstd', 'jsonl.zst', "json"]
    # for suffix in json_suffices:
    suffix = "json"
    data_files['train'] += glob.glob(f'*train*.{suffix}', root_dir=dataset_dir, recursive=True)
    data_files['validation'] += glob.glob(f'*validation*.{suffix}', root_dir=dataset_dir, recursive=True)

    data_files['train'] = sorted(data_files['train'])
    data_files['train'] = [os.path.join(dataset_dir, filename) for filename in data_files['train']]
    data_files['validation'] = sorted(data_files['validation'])
    data_files['validation'] = [os.path.join(dataset_dir, filename) for filename in data_files['validation']]
    # print(data_files['train'][0])
    # print(data_files['validation'][0])

    if dataset_subsample_rate is not None and dataset_subsample_rate < 1.0:
        data_files['train'] = uniform_sample_list(data_files['train'], dataset_subsample_rate)


    if test_split_percentage > 0.:
        # total_valid_files = max(1, int(len(data_files['train']) * data_args.validation_split_percentage))
        # stride = math.floor(len(data_files['train']) / total_valid_files)
        # data_files['test'] = copy.deepcopy(data_files['train'][::stride])

        data_files['test'] = copy.deepcopy(uniform_sample_list(data_files['train'], test_split_percentage))
        data_files['train'] = [fn for fn in data_files['train'] if fn not in data_files['test']]

    # only load one shard for a quick test
    if sanity_check:
        data_files['train'] = data_files['train'][:1]
        if len(data_files['test']) > 1:
            data_files['test'] = data_files['test'][:1]

    # remove train/test set to accelerate data loading if training/validation only
    if 'debug' in training_args.output_dir:
        data_files['validation'] = [data_files['validation'][0]]
        data_files["test"] = [data_files["test"][0]]

    if not training_args.do_train:
        data_files['train'] = []

    if not training_args.do_eval:
        data_files['validation'] = []
    
    if not training_args.do_predict:
        data_files["test"] = []

    print_rank_0(f"Loading json dataset from {dataset_dir}, {len(data_files.get('train', []))} train files, {len(data_files.get('test', []))} test files")
    raw_datasets = load_dataset("json", data_files=data_files, streaming=streaming)

    return raw_datasets

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.config_name:
        config = MyLlamaConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = MyLlamaConfig.from_pretrained(model_args.model_name_or_path)
    else:
        raise NotImplementedError

    type = model_args.config_name.split('/')[-1].split('.')[0]
    if type != config.rpe_type:
        assert config.rpe_type == 'adape', f"Not matched positional embeddding method for config file {type}.json and config content {config.rpe_type}. There's a chance you ran a method not meeting your expectations."

    # complete configuration
    scaled_max_position_embeddings=int(training_args.model_max_position_embeddings * training_args.rope_scaling_factor)
    config.max_position_embeddings=scaled_max_position_embeddings
    config.use_flash_attention_2 = training_args.use_flash_attention_2

    if training_args.rope_scaling_type is not None:
        config.rope_scaling = {
            "type": training_args.rope_scaling_type,
            "factor": training_args.rope_scaling_factor
            }
        # if 'yarn' in training_args.rope_scaling_type:
        config.original_max_position_embeddings = training_args.model_max_position_embeddings
    elif config.rpe_type in ['yarn', 'adayarn']:
        config.rope_scaling = {
            "type": config.rpe_type,
            "factor": training_args.rope_scaling_factor
            }
        config.original_max_position_embeddings = training_args.model_max_position_embeddings
        
    try:
        module_name = config.rpe_type
        MyLlamaForCausalLM = __import__(f"models.llama.{module_name}", fromlist=["MyLlamaForCausalLM"]).MyLlamaForCausalLM
    except Exception as e:
        print(e)
        rpe_types = [
            "rope", "sincos", "randrope", "alibi", "adape", "yarn", 
            "t5rb", "fire", "xpos", "nope", "adayarn", "adalibi",
        ]
        raise NotImplementedError(f"Unknown positional embedding {module_name}, choose from {rpe_types}")


    if model_args.model_name_or_path:
        model = MyLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        if training_args.local_rank == 0:
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            print(f"Finetuning model from {model_args.model_name_or_path} - Model Size={n_params/2**20:.2f}M parameters")
    else:
        model = MyLlamaForCausalLM(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        if training_args.local_rank == 0:
            print(f"Training new model from scratch - Total Size={n_params/2**20:.2f}M parameters")

    # determine if load from pretrained
    # if training_args.finetune_from_pretrained:
    #     pretrained_model = LlamaForCausalLM.from_pretrained(training_args.finetune_from_pretrained)
    #     checkpoint = pretrained_model.state_dict()
    #     def filter(key):
    #         rotary = 'sin_cached' not in key and 'cos_cached' not in key
    #         post_linear = "post_attention_linears" not in key
    #         pe_proj = "pe.proj" not in key
    #         return all((rotary, post_linear, pe_proj))
    #     filtered_checkpoint = {k: v for k, v in checkpoint.items() if filter(k)}
    #     model.load_state_dict(filtered_checkpoint, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(
        "./models/llama/llama_tokenizer",
        use_fast=True,
    )

 
    
    # raw_datasets = load_dataset("allenai/c4", "en", streaming=True)
    raw_datasets = load_json_dataset(training_args, "/scratch/gpfs/DATASETS/hugging_face/c4/en")

    def infer_columns_of_dataset(raw_datasets):
        default_cols = raw_datasets.features
    
        if default_cols is not None:
            return list(default_cols)
    
        first_example = next(iter(raw_datasets))
        if isinstance(first_example, dict):
            return list(first_example.keys())
        else:
            raise ValueError(f'Unable to infer column names from the data type: {type(first_example)}')

    if training_args.do_train:
        column_names = infer_columns_of_dataset(raw_datasets["train"])
    else:
        column_names = infer_columns_of_dataset(raw_datasets["test"])

    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        return output

    if training_args.local_rank > 0: 
        torch.distributed.barrier()

    os.makedirs(f"{data_args.dataset_cache_dir}/tokenized", exist_ok=True)
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        block_size = config.train_scale
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

        return result

    os.makedirs(f"{data_args.dataset_cache_dir}/{config.train_scale}", exist_ok=True)
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    if training_args.local_rank == 0:
        print(f"rank{training_args.local_rank} loading datasets")

    if training_args.local_rank == 0:
        print(f"rank{training_args.local_rank} datasets loaded")

    train_dataset = lm_datasets["train"]
    valid_dataset = lm_datasets["validation"]

    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    data_collator = default_data_collator # DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    data_module = dict(train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=data_collator)

    # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    #! specially for skip streaming dataset for later batch.
    n_lastest_iter = 0
    if training_args.resume_from_checkpoint == True:
        # search for the latest checkpoint
        from pathlib import Path
        all_checkpoints = list(Path(training_args.output_dir).glob("checkpoint-*"))
        all_checkpoints = [x for x in all_checkpoints if (x / "trainer_state.json").exists() and not x.name.endswith("final")]
        if len(all_checkpoints) == 0:
            training_args.resume_from_checkpoint = None
            print("No checkpoint found, starting from scratch")
        else:
            all_checkpoints = [str(x) for x in all_checkpoints]
            latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
            training_args.resume_from_checkpoint = latest_checkpoint
            print("Resuming from checkpoint", latest_checkpoint)
            n_lastest_iter = int(latest_checkpoint.split('-')[-1])

    if isinstance(train_dataset, IterableDataset):
        shuffle_seed = training_args.data_seed + n_lastest_iter if training_args.data_seed is not None else training_args.seed + n_lastest_iter
        train_dataset = train_dataset.shuffle(seed=shuffle_seed)
        training_args.ignore_data_skip = True
        print("*** Set ignore_data_skip=True for streaming mode to save time ***")


    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False

    if training_args.do_train:
        logging.info("*** Start Training ***")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_state()
        # trainer.save_model(output_dir=training_args.output_dir)
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.do_eval:
        logging.info("*** Evaluate on valid set***")
        metrics = trainer.evaluate(eval_dataset=valid_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        

if __name__ == "__main__":
    train()
