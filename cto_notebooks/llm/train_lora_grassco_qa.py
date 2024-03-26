# %%
# Setup training parameters
import glob
from itertools import permutations

from cto_notebooks.utils.config import CONFIG as SETTINGS
from cto_notebooks.utils.lora import LoraModules, LoraTrainingConfig
from cto_notebooks.utils.rag import VectorStore, average_vector

data_dir = SETTINGS.data_dir.joinpath("grascco")
raw_data_folder = data_dir.joinpath("raw")
train_data_folder = data_dir.joinpath("datasets")
label_filepath = data_dir.joinpath("grascco.json")
exclude_from_training = ["Cajal.txt", "Boeck.txt", "Baastrup.txt"]

config = LoraTrainingConfig(
    model={
        "tokenizer": "jphme/em_german_leo_mistral",
        "config": {"pretrained_model_name_or_path": "jphme/em_german_leo_mistral"},
    },
    batch_size=4,
    micro_batch_size=1,
    stop_at_loss=-1,
    epochs=1,
    output_dir=SETTINGS.cache_dir.joinpath("lora").as_posix(),
    modules=LoraModules(default=False, q=True, v=True),
    # flags=TrainingFlags(use_cpu=True)
)

train_template = {}
train_template["template_type"] = "preprocessed"

HAS_TRAIN_DATA = len(glob.glob(f"{train_data_folder}/*"))
if HAS_TRAIN_DATA:
    print(f"{train_data_folder} is not empty. Will not generate training data.")


# %%
# Create vector embedding
from typing import Dict, List, Optional, Tuple, Union

from langchain.embeddings import HuggingFaceEmbeddings

# VectorStore
chunk_size = 256
chunk_overlap = 25
vector_count = 3
vector_model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
_embeddings = HuggingFaceEmbeddings(
    model_name=vector_model_name,
    model_kwargs={"device": "cpu"},
)

# %%
# Load Grascco data
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cto_notebooks.data.grascco import load_labeled_data

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
data = load_labeled_data(label_filepath, raw_data_folder)
tests: Dict[str, Tuple[Dict[str, Union[str, List[str]]], VectorStore]] = {}
for record in data:
    tests[record["file_name"]] = (
        record,
        FAISS.from_texts(text_splitter.split_text(record["text"]), _embeddings),
    )


# %%
# Setup prompt

from typing import Dict, List

questions: List[str] = [
    "Wie heißt der Patient?",
    "Wann hat der Patient Geburstag?",
    "Wie heißt der Arzt?",
    "Wann wurde der Patient bei uns aufgenommen?",
    "Wann wurde der Patient bei uns entlassen?",
]
fields: Dict[str, str] = {
    "Wie heißt der Patient?": "patient_name",
    "Wann hat der Patient Geburstag?": "patient_date_of_birth",
    "Wie heißt der Arzt?": "attending_doctors",
    "Wann wurde der Patient bei uns aufgenommen?": "recording_date",
    "Wann wurde der Patient bei uns entlassen?": "release_date",
}

queries: Dict[str, List[float]] = {
    k: average_vector(fields[k], embeddings=_embeddings, data_dict=tests.values())
    for k in questions
}

qa_analyze_prompt = """<s>Du bist ein hilfreicher Assistent. USER: \
Kontext1: {context0} zu
Frage1: {question0} in JSON-Feld {field0}.
Kontext2: {context1} zu
Frage2: {question1} in JSON-Feld {field1}.
Kontext3: {context2} zu
Frage3: {question2} in JSON-Feld {field2}.
Kontext4: {context3} zu
Frage4: {question3} in JSON-Feld {field3}.
Kontext5: {context4} zu
Frage5: {question4} in JSON-Feld {field4}.

Gebe nur die hilfreichen Antworten unten zurück und nichts anderes. \
Halte dich außerdem sehr kurz mit der Antwort. \
ASSISTANT:{target}</s>"""  # noqa: E501


# %%
# define trainingsdata functions

import json
from copy import deepcopy

from dateparser import parse


def _format_stay_date(recording: str, release: str) -> None:
    rec_date = rel_date = None
    if release:
        tmp = parse(release, languages=["de"], date_formats=["%Y-%m-%d"])
        if tmp is None:
            msg = f"Could not parse date from {release}!"
            raise AssertionError(msg)
        rel_date = tmp.strftime("%d.%m.%Y")
    if recording:
        tmp = parse(
            recording,
            languages=["de"],
            date_formats=["%Y-%m-%d"],
            settings={"RELATIVE_BASE": tmp},
        )
        if tmp is None:
            msg = f"Could not parse date from {release}!"
            raise AssertionError(msg)
        rec_date = tmp.strftime("%d.%m.%Y")
    return rec_date, rel_date


def _format_patient_date_of_birth(
    date_str: str, target_format: str = "%d.%m.%Y"
) -> str:
    date = parse(date_str, languages=["de"], date_formats=["%Y-%m-%d"])
    return date.strftime(target_format) if date else None


def _format_patient_name(name: str) -> str:
    if name.find(", ") != -1:
        words = name.split(",")
        if len(words) == 2:
            name = words[1] + " " + words[0]
    return name.strip()


def _create_train_prompt(document: str, target: str, orig_json: Dict) -> str:
    _vectorstore: Optional[VectorStore] = None
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = text_splitter.split_text(document)
    _vectorstore = FAISS.from_texts(texts, _embeddings)

    questions_dict: Dict[str, str] = {}
    parameters: Dict[str, str] = {}
    prompt_list: List[str] = []

    parameters["target"] = target

    for question in questions:
        sorted_vecs = [
            doc.page_content
            for doc in _vectorstore.similarity_search_by_vector(
                queries[question], k=len(_vectorstore.index_to_docstore_id)
            )
        ]
        vals = orig_json[fields[question]]
        if isinstance(vals, str):
            vals = [vals]
        while not all(val in "".join(sorted_vecs[:vector_count]) for val in vals):
            sorted_vecs.pop(0)
        questions_dict[question] = sorted_vecs[:vector_count]

    for perm_patient_name in permutations(questions_dict[questions[0]]):
        parameters["context0"] = " ".join(perm_patient_name)
        parameters["question0"] = questions[0]
        parameters["field0"] = fields[questions[0]]
        for perm_patient_date_of_birth in permutations(questions_dict[questions[1]]):
            parameters["context1"] = " ".join(perm_patient_date_of_birth)
            parameters["question1"] = questions[1]
            parameters["field1"] = fields[questions[1]]
            for perm_attending_doctor in permutations(questions_dict[questions[2]]):
                parameters["context2"] = " ".join(perm_attending_doctor)
                parameters["question2"] = questions[2]
                parameters["field2"] = fields[questions[2]]
                for perm_recording_date in permutations(questions_dict[questions[3]]):
                    parameters["context3"] = " ".join(perm_recording_date)
                    parameters["question3"] = questions[3]
                    parameters["field3"] = fields[questions[3]]
                    for perm_release_date in permutations(questions_dict[questions[4]]):
                        parameters["context4"] = " ".join(perm_release_date)
                        parameters["question4"] = questions[4]
                        parameters["field4"] = fields[questions[4]]

                        prompt_list.append(qa_analyze_prompt.format(**parameters))
    return prompt_list


def _load_training_data() -> List[Dict]:
    label_dicts: List[Dict] = []
    data = load_labeled_data(label_filepath, raw_data_folder)
    for json_dict in data:
        # format names
        orig_json = deepcopy(json_dict)
        json_dict["patient_name"] = json_dict["patient_name"].replace("\n", " ")
        json_dict["attending_doctors"] = [
            doc.replace("\n", " ") for doc in json_dict["attending_doctors"]
        ]

        json_dict["patient_name"] = _format_patient_name(json_dict["patient_name"])

        # format dates
        json_dict["patient_date_of_birth"] = _format_patient_date_of_birth(
            json_dict["patient_date_of_birth"]
        )

        formatted_dates = _format_stay_date(
            json_dict["recording_date"], json_dict["release_date"]
        )

        json_dict["recording_date"] = formatted_dates[0]
        json_dict["release_date"] = formatted_dates[1]
        document = json_dict["text"]

        del json_dict["file_name"]
        del json_dict["text"]

        target = json.dumps(json_dict)

        prompt_list = _create_train_prompt(document, target, orig_json)

        for prompt in prompt_list:
            label_dict: Dict[str, str] = {}
            label_dict["target"] = target
            label_dict["instruct"] = prompt

            label_dicts.append(label_dict)

    return label_dicts


# %%
# Generate training data
from transformers import AutoTokenizer

if HAS_TRAIN_DATA:
    training_raw_data = _load_training_data()
    print(len(training_raw_data))

# %%
# Tokenize training data and save to disk

import concurrent.futures
from copy import deepcopy
from pathlib import Path

import torch.multiprocessing
from datasets import Dataset

torch.multiprocessing.set_sharing_strategy("file_system")


def tokenizer_worker(args: List) -> List[Dict[str, List]]:
    samples, start = args
    tokenizer = AutoTokenizer.from_pretrained(
        **config.model["config"], trust_remote_code=False
    )
    tokenizer.pad_token_id = 0
    tokenized = [tokenizer(s["instruct"], add_special_tokens=False) for s in samples]
    if tokenized:
        data = Dataset.from_list(tokenized)
        data.save_to_disk(f"{train_data_folder}/samples-{start}-{start+len(samples)-1}")
    return True


if not HAS_TRAIN_DATA:
    train_data = []
    n_samples = len(training_raw_data)
    n_step = 10_000

    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        for _ in executor.map(
            tokenizer_worker,
            [
                (deepcopy(training_raw_data[i : i + n_step]), i)
                for i in range(0, n_samples, n_step)
            ],
        ):
            pass

# %%
# Load training data from disk

import glob
from pathlib import Path

from datasets import Dataset, concatenate_datasets

train_data = None
for f in glob.glob(f"{train_data_folder}/*"):
    if train_data is None:
        train_data = Dataset.load_from_disk(f)
    else:
        train_data = concatenate_datasets([train_data, Dataset.load_from_disk(f)])
print(len(train_data))
# %%
# Save sample training data

tokenizer = AutoTokenizer.from_pretrained(
    **config.model["config"], trust_remote_code=False
)
tokenizer.pad_token_id = 0
# tokenizer.padding_side = "left"

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
import transformers

if "model" not in locals():
    model = transformers.AutoModelForCausalLM.from_pretrained(**config.model["config"])

# %%
# Step 5 - Setup Trainer
from pathlib import Path

from cto_notebooks.utils.training import Trainer

trainer = Trainer(config=config, model=model)
trainer.setup_training(
    train_data=train_data, train_template=train_template, tokenizer=tokenizer
)

print(f"Going to train modules: {', '.join(config.modules.target_modules(model))}")
# %%
# Step 6 - Run Trainer
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
