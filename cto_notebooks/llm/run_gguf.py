# %%
from cto_notebooks.utils.config import CONFIG

# local model
MODEL_CONFIG = {
    "model_path_or_repo_id": CONFIG.model_dir.joinpath(
        "em_german_leo_mistral_lora_cardiode_Q5_K_M.gguf"
    ).as_posix(),
    "max_new_tokens": 1024,
    "context_length": 2048,
}

# model from huggingface
# MODEL_CONFIG = {
#     "model_path_or_repo_id": "TheBloke/em_german_leo_mistral-GGUF",
#     "model_file": "em_german_leo_mistral.Q5_K_M.gguf",
#     "model_type": "mistral",
# }

# %%
from ctransformers import AutoModelForCausalLM

prompt_format = "Du bist ein hilfreicher Assistent. USER: {prompt} ASSISTANT:"
model = AutoModelForCausalLM.from_pretrained(**MODEL_CONFIG)


def query(text: str) -> str:
    return model(
        prompt_format.format(prompt=text),
        stop="</s>",
        max_new_tokens=MODEL_CONFIG["max_new_tokens"],
    )


# %%

res = query("Zusammenfassung:")
print(res)

# %%
