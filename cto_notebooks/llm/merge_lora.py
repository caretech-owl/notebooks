# %%
# Setup source and target paths

from cto_notebooks.utils.config import CONFIG

# model from huggingface
# see 'run_gguf.py' for local model
MODEL_CONFIG = {
    "pretrained_model_name_or_path": "jphme/em_german_leo_mistral",
}
# MODEL_CONFIG = {"pretrained_model_name_or_path": "LeoLM/leo-hessianai-7b-chat"}

MODEL_PATH = CONFIG.model_dir.joinpath("em_german_leo_mistral_grascco_qa")

# %%
# Step 1 - Load model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(**MODEL_CONFIG)
tokenizer = AutoTokenizer.from_pretrained(**MODEL_CONFIG)
# this should match your training padding options
tokenizer.pad_token_id = 0
# tokenizer.padding_side = "left"
# %%
# Step 2 - Merge model and LoRA
from peft import PeftModel

from cto_notebooks.utils.config import CONFIG

merged_model = PeftModel.from_pretrained(
    model, CONFIG.cache_dir.joinpath("lora")
).merge_and_unload()

# %%
# Step 3 - Save model to disk

if MODEL_PATH.exists():
    msg = (
        f"Model target path {MODEL_PATH} already exists! Change your model's name "
        "or remove the previously saved model and try again."
    )
    raise RuntimeError(msg)

merged_model.save_pretrained(MODEL_PATH, safe_serialization=True)

# %%
# Step 4 - Save tokenizer config

_ = tokenizer.save_pretrained(MODEL_PATH)

# %%
