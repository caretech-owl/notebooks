# %%
# Step 1 - Load CAS data
import os

from cto_notebooks.utils.cas import load_cas_zips

amnt = 400
raw_data = load_cas_zips(
    f"{os.environ['DATA_PATH']}/cardiode/cardiode/cas/CARDIODE400_main/", amnt
)
if len(raw_data) != amnt:
    msg = f"Expected to load {amnt} files but only got {len(raw_data)}."
    raise RuntimeError(msg)

# %%
# Convert data
from cto_notebooks.data.cardio_de import generate_data_from_cas

cas_data = [generate_data_from_cas(cas) for cas in raw_data]

# %%
# Define full text generation

from datetime import date

from faker import Faker

from cto_notebooks.data.cardio_de import CardioDeData

fake = Faker("de_DE")


def populate_fields(data: CardioDeData) -> None:
    if data.is_patientin:
        data.fields[
            "patient_name"
        ] = f"{fake.first_name_female()} {fake.last_name_female()}"
        data.fields["patient_salute"] = fake.prefix_female()
    else:
        data.fields[
            "patient_name"
        ] = f"{fake.first_name_male()} {fake.last_name_male()}"
        data.fields["patient_salute"] = fake.prefix_male()
    for field in data.fields:
        if field.startswith("address"):
            data.fields[field] = fake.address().replace("\n", ", ")
        elif field.startswith("date"):
            data.fields[field] = fake.date_between(
                date(2030, 1, 1), date(2080, 12, 31)
            ).strftime("%d.%m.%Y")
        elif field.startswith("person"):
            data.fields[field] = fake.name()
        elif field.startswith("phone"):
            data.fields[field] = fake.phone_number()
        elif field.startswith("org"):
            data.fields[field] = fake.company()
        elif field.startswith("loc"):
            data.fields[field] = fake.city()
        else:
            if field not in ["patient_name", "patient_salute"]:
                print(field)
                raise Exception()


# %%
# Generate full text
for cas in cas_data:
    populate_fields(cas)
    for sec in cas.sections:
        sec.text = sec.preprocessed.format(**cas.fields)

# %%
# Load Mixtral


import transformers

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=transformers.BitsAndBytesConfig(load_in_8bit=True),
)

# %%
# Create pipeline

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=0,
    eos_token_id=tokenizer.eos_token_id,
    device_map="auto",
)

# %%
# Define summarize method and set parameters


def summarize(long: str) -> str:
    prompt = pipeline.tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    f"Fasse bitte folgenden Text in Stichpunkten in deutscher Sprache zusammen: {long}"  # noqa: E501
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
        top_k=40,
        top_p=0.8,
        return_full_text=False,
    )
    return outputs[0]["generated_text"]


# %%
# Summarize sections
import pickle
from datetime import datetime

from cto_notebooks.data.cardio_de import SectionEnum
from cto_notebooks.utils.config import CONFIG

to_skip = {
    SectionEnum.ABSCHLUSS,
    SectionEnum.ANREDE,
    SectionEnum.BEFUNDE,
    SectionEnum.KU_BEFUNDE,
    SectionEnum.LABOR,
    SectionEnum.ECHO_BEFUNDE,
}
enumerations = {
    SectionEnum.AKTUELL_DIAGNOSEN,
    SectionEnum.AUFNAHMEMEDIKATION,
    SectionEnum.AUR,
    SectionEnum.DIAGNOSEN,
}
to_summarize = {
    SectionEnum.ANAMNESE,
    SectionEnum.ZUSAMMENFASSUNG,
}

n_sum = sum(
    1 if sec.section in to_summarize else 0 for cas in cas_data for sec in cas.sections
)

log_file_path = CONFIG.cache_dir.joinpath("logs", "cardio_summarize.log")
data_file_path = CONFIG.data_dir.joinpath("cardio_de")

for cas in cas_data:
    with open(f"{log_file_path}", "a") as f:
        print(f"{datetime.now().isoformat()}: {cas.title}", file=f)
    for sec in cas.sections:
        if sec.section in to_summarize:
            with open(f"{log_file_path}", "a") as f:
                print(f"{datetime.now().isoformat()}: Input\n{sec.text}", file=f)
            sec.summary = summarize(sec.text)
            with open(f"{log_file_path}", "a") as f:
                print(f"{datetime.now().isoformat()}: Output\n{sec.summary}", file=f)
    with open(f"{data_file_path}/{cas.title}.pkl", "wb") as f:
        pickle.dump(cas, f)
