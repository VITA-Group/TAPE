import argparse
import logging
import os

from pathlib import Path

import datasets
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset

import transformers
from transformers import (
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
)
# from model_llama import MyLlamaForCausalLM
from config_llama import MyLlamaConfig
from transformers.utils import check_min_version, send_example_telemetry


logger = get_logger(__name__)

SUMMARY_TASKS = ['summ_screen_fd', 'qmsum',  'gov_report']
OTHER_TASKS = ['narrative_qa', 'quality', "qasper", 'contract_nli']

def old_parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="polynomial",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    args = parser.parse_args()

    return args

from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
from typing import Optional
from transformers.trainer_utils import SchedulerType

@dataclass
class CustomTrainingArguments(TrainingArguments):
    model_name_or_path: Optional[str] = field(default=None)
    use_flash_attention_2: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(
        default=None, 
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={"help": "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={"help": "Optional input sequence length after tokenization. Training dataset will be truncated in blocks of this size."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for preprocessing."}
    )

    report_to: Optional[str] = field(
        default="all",
        metadata={"help": 'The integration to report results/logs to. Supported: "tensorboard", "wandb", "comet_ml", "clearml".'}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={"help": "Create the model as an empty shell, materializing parameters when pretrained weights are loaded for lower memory usage."}
    )

def parse_args():
    hfparser = HfArgumentParser(CustomTrainingArguments)
    args = hfparser.parse_args_into_dataclasses()[0]
    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("clm", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # logger.info(accelerator.state, main_process_only=False)
    if args.local_rank == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # raw_train_datasets = load_dataset('json', data_files="/mnt/bn/hzy-data-all/hierarchy_ape/passkey/train.jsonl", split="train")
    # raw_valid_datasets = load_dataset('json', data_files="/mnt/bn/hzy-data-all/hierarchy_ape/passkey/valid.jsonl", split="train")
    print(args.dataset_name)
    dataset = load_dataset(f"tau/scrolls", args.dataset_name)

    raw_train_datasets = dataset['train']
    raw_valid_datasets = dataset['validation']
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = MyLlamaConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = MyLlamaConfig.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError("check your model_name_or_path and config_name")
    config.use_flash_attention_2 = args.use_flash_attention_2

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    try:
        module_name = config.rpe_type
        MyLlamaForCausalLM = __import__(f"models.llama.{module_name}", fromlist=["MyLlamaForCausalLM"]).MyLlamaForCausalLM
    except Exception as e:
        print(e)
        exit(-1)

    if args.model_name_or_path:
        model = MyLlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            ignore_mismatched_sizes=True
        )
    else:
        model = MyLlamaForCausalLM(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    # embeddings = model.model.embeds[0].weight.data

    # Preprocessing the datasets.
    # First we tokenize all the texts.

    def tokenize_function_for_sum(examples, pad_length=8192):
        report = tokenizer("Context:\n" + examples['input'] + "\n Please summarize this report:")
        summary = tokenizer(examples['output'])
        report['input_ids'] = report['input_ids'][:7184] + report['input_ids'][-7:]  + summary['input_ids'][:1000] + [tokenizer.eos_token_id]
        report['labels'] = report['input_ids'].copy()
        report['labels'][:-(len(summary['input_ids'][:1000])+1)] = [config.pad_token_id] * len(report['labels'][:-(len(summary['input_ids'][:1000])+1)])
        # label = tokenizer(examples['label'])

        report["input_ids"] = report["input_ids"] + (pad_length - len(report["input_ids"])) * [31999]
        report["labels"] = report["labels"] + (pad_length - len(report["labels"])) * [config.pad_token_id]
        assert len(report["input_ids"]) == 8192
        assert len(report["labels"]) == 8192
        del report['attention_mask']
        assert len(report.keys()) == 2, f"{report.keys()}"
        return report

    def tokenize_function_for_qa(examples, pad_length=8192):
        report = tokenizer(" ".join(examples['input'].split(" ")[:20000]))
        summary = tokenizer(examples['output'])

        report['input_ids'] = report['input_ids'][:7991] + summary['input_ids'][:200] + [tokenizer.eos_token_id]
        report['labels'] = report['input_ids'].copy()
        report['labels'][:-(len(summary['input_ids'][:200])+1)] = [config.pad_token_id] * len(report['labels'][:-(len(summary['input_ids'][:200])+1)])
        # label = tokenizer(examples['label'])

        report["input_ids"] = report["input_ids"] + (pad_length - len(report["input_ids"])) * [31999]
        report["labels"] = report["labels"] + (pad_length - len(report["labels"])) * [config.pad_token_id]
        assert len(report["input_ids"]) == 8192
        assert len(report["labels"]) == 8192
        del report['attention_mask']
        assert len(report.keys()) == 2, f"{report.keys()}"
        return report


    tokenize_function = tokenize_function_for_sum if args.dataset_name in SUMMARY_TASKS else tokenize_function_for_qa
    tokenized_train_datasets = raw_train_datasets.map(
        tokenize_function,
        # batched=True,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on training dataset",
    )
    tokenized_valid_datasets = raw_valid_datasets.map(
        tokenize_function,
        # batched=True,
        num_proc=32,
        remove_columns=raw_valid_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)


    train_dataset = tokenized_train_datasets
    eval_dataset = tokenized_valid_datasets

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # transfer to Trainer api
    from transformers import Trainer
    args.gradient_checkpointing = True
    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=default_data_collator)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=args, **data_module)
    model.config.use_cache = False

    
    if args.resume_from_checkpoint:
        # search for the latest checkpoint
        from pathlib import Path
        all_checkpoints = list(Path(args.output_dir).glob("checkpoint-*"))
        all_checkpoints = [x for x in all_checkpoints if (x / "trainer_state.json").exists() and not x.name.endswith("final")]
        if len(all_checkpoints) == 0:
            args.resume_from_checkpoint = None
            print("No checkpoint found, starting from scratch")
        else:
            all_checkpoints = [str(x) for x in all_checkpoints]
            latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
            args.resume_from_checkpoint = latest_checkpoint
            print("Resuming from checkpoint", latest_checkpoint)
            n_lastest_iter = int(latest_checkpoint.split('-')[-1])


    print(args.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

if __name__ == "__main__":
    main()