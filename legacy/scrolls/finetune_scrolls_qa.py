import argparse
import json
import logging
import math
import os

import glob
import random
from itertools import chain
from pathlib import Path
import gc

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
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
from model_llama import MyLlamaForCausalLM
from config_llama import MyLlamaConfig
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from process import preprocess

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
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
        "--preprocess",
        type=str,
        default=None,
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
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
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
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
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

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("clm", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

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
    dataset = load_from_disk(f"/mnt/bn/hzy-data-all/hierarchy_ape/gov_report/scrolls/{args.dataset_name}")

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
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

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

    def tokenize_function(examples, pad_length=8192):
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

    with accelerator.main_process_first():
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

    # # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    #     # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    #     total_length = (total_length // block_size) * block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result

    # logger.info(f"Before group: {tokenized_datasets}")
    # with accelerator.main_process_first():
    #     os.makedirs(f"{args.dataset_cache_dir}/{args.block_size}", exist_ok=True)
    #     lm_datasets = tokenized_datasets.map(
    #         group_texts,
    #         batched=True,
    #         num_proc=args.preprocessing_num_workers,
    #         load_from_cache_file=not args.overwrite_cache,
    #         cache_file_names={"train": f"{args.dataset_cache_dir}/{args.block_size}/lm_datasets_train.arrow",\
    #          "validation": f"{args.dataset_cache_dir}/{args.block_size}/lm_datasets_validation.arrow", \
    #             "test": f"{args.dataset_cache_dir}/{args.block_size}/lm_datasets_test.arrow"},
    #         desc=f"Grouping texts in chunks of {block_size}",
    #     )
    # logger.info(f"After group: {lm_datasets}")

    train_dataset = tokenized_train_datasets
    eval_dataset = tokenized_valid_datasets

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layernorm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    else:
        num_training_steps = args.max_train_steps * accelerator.num_processes
    num_warmup_steps = args.num_warmup_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    if "sin" in config.rpe_type:
        model.model.init_sin_embeddings_new()
        model.model.pe = model.model.pe.to(accelerator.device)

    model.gradient_checkpointing_enable()
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Loader Length = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total warmup steps = {args.num_warmup_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = glob.glob(args.output_dir + "/*")
            print("dirs: ", dirs)
            # dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs = [name for name in dirs if "state" in name]
            print("checkpoint dirs: ", dirs)
            if len(dirs) != 0:
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)
            else:
                checkpoint_path = None
                args.resume_from_checkpoint = None
                
        if checkpoint_path is not None:
            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.split("_")[1]) + 1 # int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.split("_")[1]) * args.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)
                completed_steps = resume_step // args.gradient_accumulation_steps
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # min_loss = 0.0
    # patience = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        # if patience >= 500:
        #     if args.with_tracking:
        #         accelerator.end_training()  
        #     break
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                # if "ape" in config.rpe_type or "sin" in config.rpe_type:
                if "ape" in config.rpe_type:
                    train_scale = 1024
                else:
                    train_scale = None
                batch = preprocess(batch, args.preprocess, train_scale=train_scale)
                outputs = model(**batch)

                # del batch
                # torch.cuda.empty_cache()
                # gc.collect()

                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            loss = loss.detach().float()
            ppl = math.exp(loss)
            if args.with_tracking:
                accelerator.log(
                    {
                        "train_perplexity": ppl,
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=completed_steps,
                )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % 50 == 0:
                    model.eval()
                    losses = []
                    for step, batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            # if "ape" in config.rpe_type or "sin" in config.rpe_type:
                            if "ape" in config.rpe_type:
                                train_scale = 1024
                            else:
                                train_scale = None
                            batch = preprocess(batch, args.preprocess, True, train_scale=train_scale)
                            outputs = model(**batch)

                            # del batch
                            # torch.cuda.empty_cache()
                            # gc.collect()

                        loss = outputs.loss
                        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

                    losses = torch.cat(losses)
                    try:
                        eval_loss = torch.mean(losses)
                        perplexity = math.exp(eval_loss)
                    except OverflowError:
                        perplexity = float("inf")

                    torch.cuda.empty_cache()
                    gc.collect()
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "eval_perplexity": perplexity,
                                "eval_loss": eval_loss,
                                # "train_loss": total_loss.item() / len(train_dataloader),
                                "epoch": epoch,
                                "step": completed_steps,
                            },
                            step=completed_steps,
                        )
                    if args.output_dir is not None and completed_steps % args.checkpointing_steps == 0:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            f"{args.output_dir}/step_{completed_steps}", is_main_process=accelerator.is_main_process, save_function=accelerator.save
                        )
                        accelerator.save_state(f"{args.output_dir}/step_{completed_steps}_state")
                        
                        del unwrapped_model
                        torch.cuda.empty_cache()
                        gc.collect()

                    # if eval_loss <= min_loss:
                    #     patience = 0
                    #     min_loss = eval_loss
                    # else:
                    #     patience += 1
                    # if patience >= 500:
                    #     accelerator.print(f"Early stopping at epoch {epoch} and step {completed_steps}")
                    #     if args.with_tracking:
                    #         accelerator.end_training()
                    #     break
                    
            if completed_steps >= args.max_train_steps:
                if args.with_tracking:
                    accelerator.end_training()  
                break
        # output_dir = f"epoch_{epoch}_state"
        # if args.output_dir is not None:
        #     output_dir = os.path.join(args.output_dir, output_dir)
        # accelerator.save_state(output_dir)





if __name__ == "__main__":
    main()