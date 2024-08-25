from collections import namedtuple
from dataclasses import dataclass, field


@dataclass
class AdaPEConfig:
    """Stores the configuration of an [`AdaptionPromptModel`]."""

    target_modules: str = field(
        default=None, metadata={"help": "Name of the attention submodules to insert adaption prompts into."}
    )
    position_size: int = field(default=None, metadata={"help": "Size of positional embedding."})
    adapter_layers: int = field(default=None, metadata={"help": "Number of adapter layers (from the top)"})

    # def __post_init__(self):
    #     self.peft_type = PeftType.ADAPTION_PROMPT

    # @property
    # def is_adaption_prompt(self) -> bool:
    #     """Return True if this is an adaption prompt config."""
    #     return True
