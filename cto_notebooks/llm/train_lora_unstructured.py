# %%
## Setup training
from cto_notebooks.utils.config import CONFIG as SETTINGS
from cto_notebooks.utils.lora import LoraModules, LoraTrainingConfig

config = LoraTrainingConfig(
    model={
        "tokenizer": "jphme/em_german_leo_mistral",
        "config": {"pretrained_model_name_or_path": "jphme/em_german_leo_mistral"},
    },
    output_dir=SETTINGS.cache_dir.joinpath("lora").as_posix(),
    # flags=TrainingFlags(use_cpu=True)
    modules=LoraModules(default=False, q=True, v=True),
)

train_template = {}
train_template["template_type"] = "raw_text"

# %%
# Step 1 - Load CAS data
from cto_notebooks.utils.cas import load_cas_zips

amnt = 400
data = load_cas_zips("/home/neum_al/data/cardio/cardiode/cas/CARDIODE400_main/", amnt)
if len(data) != amnt:
    msg = f"Expected to load {amnt} files but only got {len(data)}."
    raise RuntimeError(msg)

# %%
# Step 2 - Remove spacey artifacts and whitespaces
import re

from cto_notebooks.utils.data import despacyfy

training_texts = []
for cas in data:
    secs = cas.select("webanno.custom.Sectionsentence")
    for i in range(len(secs)):
        if secs[i].Sectiontypes in ["Anamnese", "Zusammenfassung"]:
            text = despacyfy(
                cas.get_sofa().sofaString[secs[i].begin : secs[i + 1].begin]
            )
            text = re.sub(r"<\[Pseudo\] ([^\>]+)>", r"\1", text)
            training_texts.append(text)

# %%
# Step 3 - Raw text to tokenized Dataset

import json
from pathlib import Path

import transformers
from datasets import Dataset

from cto_notebooks.utils.data import split_chunks, tokenize

tokenizer = transformers.AutoTokenizer.from_pretrained(
    config.model["tokenizer"], trust_remote_code=False, use_fast=True
)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

training_tokens = []
for text in training_texts:
    if len(text) == 0:
        continue
    training_tokens.extend(
        split_chunks(
            tokenizer.encode(text),
            config.cutoff_len,
            config.cutoff_len - config.overlap_len,
        )
    )
text_chunks = [tokenizer.decode(x) for x in training_tokens]
train_data = Dataset.from_list(
    [tokenize(x, tokenizer, config.cutoff_len) for x in text_chunks]
)

decoded_entries = []
for i in range(min(10, len(train_data))):
    decoded_text = tokenizer.decode(train_data[i]["input_ids"])
    decoded_entries.append({"value": decoded_text})

# Write the log file
log_dir = SETTINGS.cache_dir.joinpath("logs")
log_dir.mkdir(exist_ok=True)
with open(Path(f"{log_dir}/train_dataset_sample.json"), "w") as json_file:
    json.dump(decoded_entries, json_file, indent=4)

# %%
# Step 4 - Load model
import time

if "model" not in locals():
    model = transformers.AutoModelForCausalLM.from_pretrained(**config.model["config"])
# %%
# Step 5 - Setup Trainer
from pathlib import Path

from cto_notebooks.utils.training import Trainer

if Path(config.output_dir).joinpath("adapter_model.safetensors").exists():
    msg = (
        f"LoRA target directory {config.output_dir}"
        " must not contain another lora adapter."
    )
    raise AssertionError(msg)

trainer = Trainer(config=config, model=model)
trainer.setup_training(
    train_data=train_data, train_template=train_template, tokenizer=tokenizer
)

print(f"Going to train modules: {', '.join(config.modules.target_modules(model))}")
# %%
## Step 6 - Run Trainer
# transformers.logging.set_verbosity_info()
thread = trainer.train()
start_time = time.perf_counter()
last_step = 0

print("Training started...")
while thread.is_alive():
    time.sleep(0.5)
    if trainer.tracked.interrupted:
        print(
            "Interrupting, please wait... "
            "*(Run will stop after the current training step completes.)*"
        )

# Saving in the train thread might fail if an error occurs, so save here if so.
if not trainer.tracked.did_save:
    trainer.save()

if trainer.tracked.interrupted:
    print(f"Interrupted. Incomplete LoRA saved to `{config.output_dir}`.")
else:
    print(
        f"Done! LoRA saved to `{config.output_dir}`.\n\nBefore testing your new LoRA, "
        "make sure to first reload the model, as it is currently dirty from training."
    )

# %%
