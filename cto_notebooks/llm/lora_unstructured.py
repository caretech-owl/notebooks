# %%
from cto_notebooks.utils.config import CONFIG as SETTINGS
from cto_notebooks.utils.lora import MODEL_CLASSES, LoraTrainingConfig, TrainingFlags

config = LoraTrainingConfig(
    model={
        "tokenizer": "jphme/em_german_leo_mistral",
        "config": {"pretrained_model_name_or_path": "jphme/em_german_leo_mistral"},
    },
    # flags=TrainingFlags(use_cpu=True)
)

lora_dir = SETTINGS.cache_dir.joinpath("lora")
WANT_INTERRUPT: bool = False


train_template = {}
train_template["template_type"] = "raw_text"


# %% [markdown]
## Convert CAS data (texts) into dataset

# %%
# Step 1 - Load CAS data
from cto_notebooks.utils.cas import load_cas_zips

amnt = 5
data = load_cas_zips("/home/neum_al/data/cardio/cardiode/cas/CARDIODE400_main/", amnt)
if len(data) != amnt:
    msg = f"Expected to load 400 files but only got {len(data)}."
    raise RuntimeError(msg)

# %%
# Step 2 - Remove spacey artifacts and whitespaces
import re

training_texts = []
for cas in data:
    secs = cas.select("webanno.custom.Sectionsentence")
    for i in range(len(secs)):
        if secs[i].Sectiontypes in ["Anamnese", "Zusammenfassung"]:
            text = cas.get_sofa().sofaString[secs[i].begin : secs[i + 1].begin]
            text = (
                text.replace("-RRB-", ")")
                .replace("-LRB-", "(")
                .replace("-UNK-", "-")
                .replace("( ", "(")
                .replace(" )", ")")
                .replace("  ", " ")
            )
            text = re.sub(r"<\[Pseudo\] ([^\>]+)>", r"\1", text)
            check = re.findall(r"-(RRB|UNK|LRB)-", text)
            if len(check) != 0:
                msg = f"Did not expect to find {check} in\n{text}."
                raise RuntimeError(msg)
            # check = re.findall(r'(B|I)-(PER|SALUTE)', text)
            # assert len(check) == 0, f"{check}\n{text}"
            training_texts.append(text)

# %%
# Step 3 - Raw text to tokenized Dataset

import threading

import torch
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

# %%
# Step 4 - Load model
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

if "model" not in locals():
    model = transformers.AutoModelForCausalLM.from_pretrained(**config.model["config"])
model_type = type(model).__name__
model_id = MODEL_CLASSES[model_type]

# %%
# Step 5 - Train LoRA
from typing import Dict, List, Tuple

from cto_notebooks.utils.lora import LLMModelProto


def calc_trainable_parameters(model: LLMModelProto) -> Tuple[List, List]:
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


if (
    not hasattr(model, "lm_head") or hasattr(model.lm_head, "weight")
) and "quantization_config" in model.config.to_dict():
    prepare_model_for_kbit_training(model)


training_config = LoraConfig(
    lora_alpha=config.lora_alpha,
    target_modules=config.modules.target_modules(model),
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

if lora_dir.joinpath("adapter_model.safetensors").exists():
    msg = (
        f"LoRA target directory {lora_dir.as_posix()}"
        " must not contain another lora adapter."
    )
    raise AssertionError(msg)

model_trainable_params, model_all_params = calc_trainable_parameters(model)
lora_model = get_peft_model(model, training_config)


class Tracked:
    def __init__(self) -> None:
        self.current_steps = 0
        self.max_steps = 0
        self.did_save = False


tracked = Tracked()
gradient_accumulation_steps = config.batch_size // config.micro_batch_size
actual_save_steps = math.ceil(config.save_steps / gradient_accumulation_steps)
projections_string = ", ".join(
    [
        projection.replace("_proj", "")
        for projection in config.modules.target_modules(model)
    ]
)
actual_lr = float(config.learning_rate)

train_log = {}


class Callbacks(transformers.TrainerCallback):
    def on_step_begin(
        self,
        args: transformers.TrainingArguments,  # noqa: ARG002
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: int,  # noqa: ARG002
    ) -> None:
        tracked.current_steps = state.global_step * gradient_accumulation_steps
        tracked.max_steps = state.max_steps * gradient_accumulation_steps
        if WANT_INTERRUPT:
            control.should_epoch_stop = True
            control.should_training_stop = True
        elif (
            state.global_step > 0
            and actual_save_steps > 0
            and state.global_step % actual_save_steps == 0
        ):
            lora_model.save_pretrained(
                f"{lora_dir}/checkpoint-{tracked.current_steps}/"
            )
            # Save log
            with open(
                f"{lora_dir}/checkpoint-{tracked.current_steps}/training_log.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(train_log, file, indent=2)
            # == Save training prompt ==
            with open(
                f"{lora_dir}/checkpoint-{tracked.current_steps}/training_prompt.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(train_template, file, indent=2)

    def on_substep_end(
        self,
        args: transformers.TrainingArguments,  # noqa: ARG002
        state: transformers.TrainerState,  # noqa: ARG002
        control: transformers.TrainerControl,
        **kwargs: int,  # noqa: ARG002
    ) -> None:
        tracked.current_steps += 1
        if WANT_INTERRUPT:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_log(
        self,
        args: transformers.TrainingArguments,  # noqa: ARG002
        state: transformers.TrainerState,  # noqa: ARG002
        control: transformers.TrainerControl,
        logs: Dict,
        **kwargs: int,  # noqa: ARG002
    ) -> None:
        train_log.update(logs)
        train_log.update({"current_steps": tracked.current_steps})
        if WANT_INTERRUPT:
            print("\033[1;31;1mInterrupted by user\033[0;37;0m")

        print(f"\033[1;30;40mStep: {tracked.current_steps} \033[0;37;0m", end="")
        if "loss" in logs:
            loss = float(logs["loss"])
            if loss <= config.stop_at_loss:
                control.should_epoch_stop = True
                control.should_training_stop = True
                print(
                    f"\033[1;31;1mStop Loss {config.stop_at_loss} reached.\033[0;37;0m"
                )


# Fix training for mixed precision models
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()

trainer = transformers.Trainer(
    model=lora_model,
    train_dataset=train_data,
    eval_dataset=None,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=math.ceil(config.warmup_steps / gradient_accumulation_steps),
        num_train_epochs=config.epochs,
        learning_rate=actual_lr,
        fp16=not (config.flags.use_cpu or config.flags.use_bf16),
        bf16=config.flags.use_bf16,
        optim=config.optimizer,
        logging_steps=2 if config.stop_at_loss > 0 else 5,
        evaluation_strategy="no",
        eval_steps=None,
        save_strategy="no",
        output_dir=lora_dir,
        lr_scheduler_type=config.lr_scheduler,
        load_best_model_at_end=False,
        # TODO: Enable multi-device support
        ddp_find_unused_parameters=None,
        no_cuda=config.flags.use_cpu,
        use_ipex=config.flags.use_ipex,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[Callbacks()],
)

lora_model.config.use_cache = False

if torch.__version__ >= "2" and sys.platform != "win32":
    lora_model = torch.compile(lora_model)

# == Save parameters for reuse ==
with open(f"{lora_dir}/training_parameters.json", "w", encoding="utf-8") as file:
    json.dump(asdict(config), file, indent=2)

# == Save training prompt ==
with open(f"{lora_dir}/training_prompt.json", "w", encoding="utf-8") as file:
    json.dump(train_template, file, indent=2)

# == Main run and monitor loop ==
lora_trainable_param, lora_all_param = calc_trainable_parameters(lora_model)

projections_string = ", ".join(
    [
        projection.replace("_proj", "")
        for projection in config.modules.target_modules(model)
    ]
)

print(f"Training '{model_id}' model using ({projections_string}) projections")

if lora_all_param > 0:
    train_percentage = 100 * lora_trainable_param / lora_all_param
    print(
        f"Trainable params: {lora_trainable_param:,d} ({train_percentage:.4f} %),"
        f"All params: {lora_all_param:,d} (Model: {model_all_params:,d})"
    )

train_log.update({"base_model_config": config.model})
train_log.update({"base_model_class": model.__class__.__name__})
train_log.update(
    {"base_loaded_in_4bit": getattr(lora_model, "is_loaded_in_4bit", False)}
)
train_log.update(
    {"base_loaded_in_8bit": getattr(lora_model, "is_loaded_in_8bit", False)}
)
train_log.update({"projections": projections_string})

if config.stop_at_loss > 0:
    print(
        f"Monitoring loss \033[1;31;1m(Auto-Stop at: {config.stop_at_loss})\033[0;37;0m"
    )

if WANT_INTERRUPT:
    print("Interrupted before start.")
    raise InterruptedError()


def log_train_dataset(trainer: transformers.Trainer) -> None:
    decoded_entries = []
    # Try to decode the entries and write the log file
    # Iterate over the first 10 elements in the dataset
    # (or fewer if there are less than 10)
    for i in range(min(10, len(trainer.train_dataset))):
        decoded_text = tokenizer.decode(trainer.train_dataset[i]["input_ids"])
        decoded_entries.append({"value": decoded_text})

    # Write the log file
    Path("logs").mkdir(exist_ok=True)
    with open(Path("logs/train_dataset_sample.json"), "w") as json_file:
        json.dump(decoded_entries, json_file, indent=4)


def threaded_run() -> None:
    log_train_dataset(trainer)
    trainer.train()
    # Note: save in the thread in case the gradio thread breaks (eg browser closed)
    lora_model.save_pretrained(lora_dir)
    # Save log
    with open(f"{lora_dir}/training_log.json", "w", encoding="utf-8") as file:
        json.dump(train_log, file, indent=2)


thread = threading.Thread(target=threaded_run)
thread.start()
last_step = 0
start_time = time.perf_counter()

while thread.is_alive():
    time.sleep(0.5)
    if WANT_INTERRUPT:
        print(
            "Interrupting, please wait... "
            "*(Run will stop after the current training step completes.)*"
        )

    elif tracked.current_steps != last_step:
        last_step = tracked.current_steps
        time_elapsed = time.perf_counter() - start_time
        if time_elapsed <= 0:
            timer_info = ""
            total_time_estimate = 999
        else:
            its = tracked.current_steps / time_elapsed
            timer_info = f"`{its:.2f}` it/s" if its > 1 else f"`{1.0 / its:.2f}` s/it"
            total_time_estimate = (1.0 / its) * (tracked.max_steps)

        # print(f"Running... **{tracked.current_steps}** / **{tracked.max_steps}** ...
        # {timer_info}, {format_time(time_elapsed)}
        # / {format_time(total_time_estimate)} ...
        # {format_time(total_time_estimate - time_elapsed)} remaining")

# Saving in the train thread might fail if an error occurs, so save here if so.
if not tracked.did_save:
    lora_model.save_pretrained(lora_dir)

if WANT_INTERRUPT:
    print(f"Interrupted. Incomplete LoRA saved to `{lora_dir}`.")
else:
    print(
        f"Done! LoRA saved to `{lora_dir}`.\n\nBefore testing your new LoRA, make sure"
        "to first reload the model, as it is currently dirty from training."
    )
