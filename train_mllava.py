from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
import torch
import os
import wandb
from utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from models.conversation import conv_mllava_v1 as default_conv, conv_templates
from utils import load_data_from_config, set_default_image_token, set_default_image_token_id, set_ignore_index
from pathlib import Path
from typing import Optional
from pathlib import Path
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
IGNORE_INDEX = -100
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
import sys

# 获取当前文件所在的目录以及上级目录
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
# 将目录添加到 sys.path
sys.path.append(current_directory)
sys.path.append(parent_directory)
# print("sys.path:", sys.path)

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# to avoid the cu10 error
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

@dataclass
class DataArguments:
    max_seq_len: Optional[int] = field(
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "
                          "than this will be truncated.", "default": 1024, "required": False},
        default=1024,
    )
    data_config_file: Optional[str] = field(
        metadata={"help": "Pretrained config name or path if not the same as model_name", "default": None, "required": False},
        default=None,
    )
    dataset_balancing: Optional[bool] = field(
        metadata={"help": "Whether to balance the dataset", "default": True, "required": False},
        default=True,
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models", "default": "llava-hf/llava-1.5-7b-hf", "required": False},
        default="llava-hf/llava-1.5-7b-hf",
    )
    tuner_type: Optional[str] = field(
        metadata={"help": "Which tuning method to use", "required": False},
        default='lora'
    )
    # lora_enabled: Optional[bool] = field(
    #     metadata={"help": "Whether to use LoRA", "default": False, "required": False},
    #     default=False,
    # )
    qlora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use QLoRA", "default": False, "required": False},
        default=False,
    )
    dora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use Dora", "default": False, "required": False},
        default=True,
    )
    position_size: Optional[int] = field(
        metadata={"help": "Positional embedding size used in adape"},
        default=32,
    )
    lora_r: Optional[int] = field(
        metadata={"help": "LoRA r", "default": 128, "required": False},
        default=128,
    )
    lora_alpha: Optional[float] = field(
        metadata={"help": "LoRA alpha", "default": 256, "required": False},
        default=256,
    )
    lora_dropout: Optional[float] = field(
        metadata={"help": "LoRA dropout", "default": 0.05, "required": False},
        default=0.05,
    )
    lora_bias: Optional[str] = field(
        metadata={"help": "LoRA bias", "default": 'none', "required": False},
        default='none',
    )
    attn_implementation: Optional[str] = field(
        metadata={"help": "The attention implementation to use", "default": "flash_attention_2", "required": False},
        default="flash_attention_2",
    )
    max_image_size: Optional[str] = field(
        metadata={"help": "The maximum image size", "default": "(1080,1920)", "required": False},
        default="(1080,1920)",
    )
    mllava_type: Optional[str] = field(
        metadata={"help": "The type of mllava model to use. ['llava', 'mllava', 'llava_next', 'mllava_next']", "default": "llava", "required": False},
        default="llava",
    )
    conv_template : Optional[str] = field(
        metadata={"help": "The conversation template to use", "default": None, "required": False},
        default=None,
    )
    projector : Optional[str] = field(
        metadata={"help": "The projector from vision to LLM", "default": "MLP", "required": False},
        default="MLP",
    )
    
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_model(model_args, training_args):
    print("Loading model...")
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    
    if model_args.qlora_enabled:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=torch_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["vision_tower"],
        )
    else:
        bnb_config = None
        
    if model_args.mllava_type == "llava":
        from models.mllava import LlavaForConditionalGeneration, MLlavaProcessor, LlavaConfig
        processor = MLlavaProcessor.from_pretrained(model_args.model_name_or_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch_dtype, 
            attn_implementation = model_args.attn_implementation,
            quantization_config=bnb_config if model_args.qlora_enabled else None,
        )
        print("Successfully loaded model from:", model_args.model_name_or_path)
    else:
        raise ValueError("Invalid mllava type")

    # keep the vision backbone frozen all the time
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False

    if bnb_config:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # print(model.num_parameters() / 1024 / 1024 / 1024)
    if model_args.tuner_type == 'lora':
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
            use_dora=model_args.dora_enabled,
        )
        print("Adding LoRA adapters...")
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

    elif model_args.tuner_type == 'adape':
        def find_attention_mlp_layers():
            return []
        from models.adape import AdaPEConfig, get_adape_model
        adape_config = AdaPEConfig(
            position_size=model_args.position_size,
            target_modules=find_attention_mlp_layers(),
        )
        model.enable_input_require_grads()
        model.language_model = get_adape_model(model.language_model, adape_config)

    elif model_args.tuner_type == 'llama_adapter':
        from peft.tuners.adaption_prompt import AdaptionPromptConfig
        config = AdaptionPromptConfig(
           target_modules=find_all_linear_names(model),
           adapter_len=3,
           task_type="CAUSAL_LM", 
        )
        model = get_peft_model(model, config)

    print(model.num_parameters() / 1024 / 1024 / 1024)
    print("Successfully added adapters")

    set_default_image_token("<image>")
    set_default_image_token_id(processor.tokenizer.convert_tokens_to_ids("<image>"))
    set_ignore_index(IGNORE_INDEX)
    return model, processor

def main(
    training_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
):
    training_args.output_dir = Path(training_args.output_dir) / model_args.model_name_or_path.split("/")[-1] / training_args.run_name
    
    training_args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args.output_dir = str(training_args.output_dir)
    training_args.remove_unused_columns = False
    data_args.is_master_worker = training_args.local_rank in [-1, 0]
    
    if not training_args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = True
    if training_args.resume_from_checkpoint == True:
        # search for the latest checkpoint
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
    
    model, processor = load_model(model_args, training_args)
    
    if model_args.conv_template:
        data_args.conv_format = conv_templates[model_args.conv_template] 
    else:
        if "llama-3" in model.language_model.name_or_path.lower():
            data_args.conv_format = conv_templates['llama_3']
        else:
            data_args.conv_format = default_conv
    print("Using conversation template:", data_args.conv_format)
    if data_args.data_config_file is not None:
        train_dataset, val_dataset, test_dataset, collate_fn = load_data_from_config(data_args, processor)
    else:
        raise ValueError("Data config file is required")
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor
    )
    if trainer.is_world_process_zero():
        print("Training arguments:")
        print(training_args)
        print("Data arguments:")
        print(data_args)
        print("Model arguments:")
        print(model_args)
    if training_args.do_train:
        print("Training model...")
        print("num_parameters: ", model.num_parameters() / 1024 / 1024 / 1024)
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # save
        final_checkpoint_dir = os.path.join(training_args.output_dir, 'checkpoint-final')
        if model_args.tuner_type in ['lora', 'llama_adapter']:
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), model_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                model.config.save_pretrained(final_checkpoint_dir)
                model.save_pretrained(final_checkpoint_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(final_checkpoint_dir, 'non_lora_trainables.bin'))
        else:
            trainer.save_model(output_dir=final_checkpoint_dir)
        processor.save_pretrained(final_checkpoint_dir)
    if training_args.do_predict:
        print("Predicting...")
        trainer.predict(test_dataset)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    print("Training Arguments: ", training_args)
    print("Data Arguments: ", data_args)
    print("Model Arguments: ", model_args)
    main(training_args, data_args, model_args)
