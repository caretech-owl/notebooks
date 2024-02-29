# %%
# Setup inference

# model from huggingface
# see 'run_gguf.py' for local model
MODEL_CONFIG = {
    "pretrained_model_name_or_path": "jphme/em_german_leo_mistral",
}
# %%
# Step 1 - Load model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(**MODEL_CONFIG)
# %%
# Step 2 - Merge model and LoRA
from peft import PeftModel

from cto_notebooks.utils.config import CONFIG

merged_model = PeftModel.from_pretrained(
    model, CONFIG.cache_dir.joinpath("lora")
).merge_and_unload()

# %%
# Step 3 - Update model config

merged_model.config.pad_token_id = 0
merged_model.config.padding_side = "left"

# %%
# Step 4 - Save to disk

model_path = CONFIG.model_dir.joinpath("em_german_leo_mistral_lora_cardiode")
if model_path.exists():
    msg = (
        f"Model target path {model_path} already exists! Change your model's name "
        "or remove the previously saved model and try again."
    )
    raise RuntimeError(msg)

merged_model.save_pretrained(model_path)

# %%
