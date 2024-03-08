# %%
# Setup training
import torch
from transformers import BitsAndBytesConfig

from cto_notebooks.utils.config import CONFIG as SETTINGS
from cto_notebooks.utils.lora import LoraModules, LoraTrainingConfig

config = LoraTrainingConfig(
    model={
        "tokenizer": "jphme/em_german_leo_mistral",
        "config": {
            "pretrained_model_name_or_path": "jphme/em_german_leo_mistral",
            "device_map": "auto",
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            ),
        },
    },
    optimizer="adamw_bnb_8bit",
    output_dir=SETTINGS.cache_dir.joinpath("lora").as_posix(),
    batch_size=4,
    micro_batch_size=1,
    epochs=3,
    cutoff_len=4096,
    modules=LoraModules(default=False, q=True, v=True),
)

train_template = {}
train_template["template_type"] = "dataset"

# %%
# Load CAS data
import glob
import pickle
from collections import Counter

from cto_notebooks.data.cardio_de import CardioDeData, CardioDeSection, SectionEnum

data = []
for file_path in glob.glob("../data/cas-*.pkl"):
    with open(file_path, "rb") as f:
        data.append(pickle.load(f))  # noqa: S301

# Filter data
#
filtered = [
    cas
    for cas in data
    if Counter(cas.section_list).most_common(1)[0][1] == 1
    and SectionEnum.ANAMNESE in cas.section_list
    and SectionEnum.ZUSAMMENFASSUNG in cas.section_list
    and SectionEnum.ABSCHLUSS in cas.section_list
    and SectionEnum.ENTLASSMEDIKATION in cas.section_list
    and SectionEnum.DIAGNOSEN in cas.section_list
    and "address_0" in cas.fields
    and len(
        next(sec for sec in cas.sections if sec.section == SectionEnum.ANAMNESE).summary
    )
    > 100
]
len(filtered)
# %%
# Define template
template = """Du bist ein hilfreicher Assistent. USER:\
-- Name: {patient_salute} {patient_name}
-- Geburtsdatum: {patient_date_of_birth}
-- Adresse: {patient_address}
-- Aufnahmedatum: {recording_date}
-- Entlassdatum: {release_date}
-- Behandelnder Arzt: {attending_doctor}
-- Anamnese:
{anamnesis_summary}
-- Befunde:
{reports_text}
-- Diagnosen:
{diagnoses_text}
-- Zusammenfassung:
{summary_summary}
-- Medikation
{medication_text}\
ASSISTANT:\
Sehr geehrte Kollegen,

{patient_salute} {patient_name}, geboren am {patient_date_of_birth}, wohnhaft in {patient_address}, befand sich vom {recording_date} bis {release_date} in unserer stationären Behandlung. Die Ergebnisse unserer Anamnese und Behandlung und weiterführende Therapieempfehlungen entnehmen Sie bitte unseren folgenden Ausführungen.

{anamnesis_text}

{reports_text}

{diagnoses_text}

{summary_text}

{medication_text}

Selbstverständlich können Präparate mit gleichem Wirkstoff und gleicher Wirkung von anderen Herstellern verordnet werden. Ebenso ist es natürlich möglich, verwandte Wirkstoffe aus derselben Wirkstoffgruppe alternativ zu verordnen.

Wir danken für die vertrauensvolle Zusammenarbeit und stehen bei Rückfragen selbstverständlich jederzeit gerne zur Verfügung.

Mit freundlichen Grüßen
{attending_doctor}
Assistenzarzt
"""  # noqa: E501

# %%
# Define template fill strategy

from datetime import date

from faker import Faker

fake = Faker("de_DE")


def fill_template(cas: CardioDeData) -> str:
    sec_anamnesis = next(
        sec for sec in cas.sections if sec.section == SectionEnum.ANAMNESE
    )
    sec_befunde = [
        sec
        for sec in cas.sections
        if sec.section
        in [SectionEnum.KU_BEFUNDE, SectionEnum.BEFUNDE, SectionEnum.ECHO_BEFUNDE]
    ]
    sec_summary = next(
        sec for sec in cas.sections if sec.section == SectionEnum.ZUSAMMENFASSUNG
    )
    sec_diagnoses = next(
        sec for sec in cas.sections if sec.section == SectionEnum.DIAGNOSEN
    )
    sec_medication = next(
        sec for sec in cas.sections if sec.section == SectionEnum.ENTLASSMEDIKATION
    )

    return template.format(
        patient_salute=cas.fields["patient_salute"],
        patient_name=cas.fields["patient_name"],
        patient_date_of_birth=fake.date_between(
            date(1960, 1, 1), date(2010, 12, 31)
        ).strftime("%d.%m.%Y"),
        patient_address=cas.fields["address_0"],
        recording_date=fake.date_between(date(2030, 1, 1), date(2080, 12, 31)).strftime(
            "%d.%m.%Y"
        ),
        release_date=fake.date_between(date(2030, 1, 1), date(2080, 12, 31)).strftime(
            "%d.%m.%Y"
        ),
        attending_doctor=fake.name(),
        anamnesis_summary=sec_anamnesis.summary.strip().replace("* ", "- "),
        anamnesis_text=sec_anamnesis.text,
        reports_text="\n".join(sec.text for sec in sec_befunde),
        diagnoses_text=sec_diagnoses.text,
        summary_summary=sec_summary.summary.strip().replace("* ", "- "),
        summary_text=sec_summary.text,
        medication_text=sec_medication.text,
    )


# %%
# Tokenize training data

from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    **config.model["config"], trust_remote_code=False
)
tokenizer.pad_token_id = 0

train_data = Dataset.from_list([tokenizer(fill_template(entry)) for entry in filtered])

# %%
# Save sample training data
import json

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
# Load model
import transformers

if "model" not in locals():
    model = transformers.AutoModelForCausalLM.from_pretrained(**config.model["config"])

# %%
# Setup Trainer
from pathlib import Path

from cto_notebooks.utils.training import Trainer

trainer = Trainer(config=config, model=model)
trainer.setup_training(
    train_data=train_data, train_template=train_template, tokenizer=tokenizer
)

print(f"Going to train modules: {', '.join(config.modules.target_modules(model))}")
# %%
# Run Trainer
import time

if Path(config.output_dir).joinpath("adapter_model.safetensors").exists():
    msg = (
        f"LoRA target directory {config.output_dir}"
        " must not contain another lora adapter."
    )
    raise AssertionError(msg)

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
