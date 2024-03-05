from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Protocol

# from peft.utils.other import (
#     TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as model_to_lora_modules,
# )
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

MODEL_CLASSES = {v[1]: v[0] for v in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.items()}


class LLMModelProto(Protocol):
    def named_modules(self) -> Dict[str, List]:
        pass


@dataclass
class LoraModules:
    q: Optional[bool] = None
    v: Optional[bool] = None
    k: Optional[bool] = None
    o: Optional[bool] = None
    gate: Optional[bool] = None
    down: Optional[bool] = None
    up: Optional[bool] = None
    default: bool = True

    def target_modules(self, model: LLMModelProto) -> List[str]:
        avail = find_target_modules(model)
        return [
            f"{name}_proj"
            for name, enabled in asdict(self).items()
            if (enabled is True or (enabled is None and self.default is True))
            and f"{name}_proj" in avail
        ]


@dataclass
class TrainingFlags:
    use_cpu: bool = False
    use_bf16: bool = False
    use_ipex: bool = False
    # True if is_torch_xpu_available() and not USE_CPU else False


@dataclass
class LoraTrainingConfig:
    model: Dict[str, str]
    output_dir: str

    cutoff_len: int = 256
    overlap_len: int = 128
    train_only_after: str = ""
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    save_steps: int = 500  # Save every n steps
    warmup_steps: int = 100
    epochs: int = 3
    r: int = 8
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # learning_rate
    # 3e-4 is a good starting base point.
    # 1e-2 is extremely high, 1e-6 is extremely low.
    learning_rate: float = 3e-4
    batch_size: int = 128
    micro_batch_size: int = 4
    stop_at_loss: float = 0  # (reasonable numbers are 1.5-1.8)

    # optimizer = [
    #     'adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla',
    #     'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision',
    #     'sgd', 'adagrad']
    # Different optimizer implementation options, for advanced users.
    # Effects of different options are not well documented yet.',
    optimizer: str = "adamw_torch"

    # lr_scheduler = [
    #    'linear', 'constant', 'constant_with_warmup', 'cosine',
    #    'cosine_with_restarts', 'polynomial', 'inverse_sqrt']
    # Learning rate scheduler - defines how the learning rate changes over time.
    # "Constant" means never change, "linear" means to go in a straight line from the
    # learning rate down to 0, cosine follows a curve, etc.'
    lr_scheduler: str = "linear"

    modules: LoraModules = field(default_factory=LoraModules)
    flags: TrainingFlags = field(default_factory=TrainingFlags)

    def __post_init__(self) -> None:
        if self.cutoff_len <= self.overlap_len:
            msg = (
                "Overlap must be smaller than cutoff"
                f"({self.cutoff_len}) but is {self.overlap_len}"
            )
            raise ValueError(msg)


def find_target_modules(model: LLMModelProto) -> List[str]:
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split(".")[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)
