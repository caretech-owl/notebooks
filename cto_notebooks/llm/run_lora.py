# %%
# Setup inference
from cto_notebooks.utils.config import CONFIG

# model from huggingface
# see 'run_gguf.py' for local model
MODEL_CONFIG = {
    "pretrained_model_name_or_path": "jphme/em_german_leo_mistral",
    "device_map": "auto",
}

PIPELINE_CONFIG = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "repetition_penalty": 1.2,
}


# %%
# Step 1 - Load model and tokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(**MODEL_CONFIG)
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(**MODEL_CONFIG)
# Make sure this matches your LoRA training
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
# %%
# Step 2 - Load and enable LoRA

# make sure you trained a LoRA first
# alternatively you can download one and put the folder in your cache
model.load_adapter(CONFIG.cache_dir.joinpath("lora"), adapter_name="lora")
model.enable_adapters()

# %%
# Step 3 - Setup and wrap pipeline

from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    device_map="auto",
    framework="pt",
)

# This should match the model and your LoRA
prompt_format = "Du bist ein hilfreicher Assistent. USER: {prompt} ASSISTANT:"


def query(text: str) -> str:
    res = pipe(prompt_format.format(prompt=text), **PIPELINE_CONFIG)
    return res[0]["generated_text"]


# %%
# Step 4 - Run inference

# Make sure you pass text that matches your LoRA training
print(query("Zusammenfassung:"))
# %%
