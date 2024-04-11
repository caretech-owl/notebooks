# %%
# Setup training parameters
import glob

from cto_notebooks.utils.config import CONFIG as SETTINGS
from cto_notebooks.utils.lora import LoraTrainingConfig

data_dir = SETTINGS.data_dir.joinpath("grascco")
raw_data_folder = data_dir.joinpath("raw")
train_data_folder = data_dir.joinpath("datasets")
label_filepath = data_dir.joinpath("grascco.json")
exclude_from_training = ["Cajal.txt", "Boeck.txt"]

config = LoraTrainingConfig(
    model={
        "tokenizer": "DiscoResearch/DiscoLM_German_7b_v1",
        "config": {
            "pretrained_model_name_or_path": "DiscoResearch/DiscoLM_German_7b_v1"
        },
    },
    stop_at_loss=-1,
    epochs=5,
    batch_size=64,
    micro_batch_size=2,
    output_dir=SETTINGS.cache_dir.joinpath("lora").as_posix(),
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
from cto_notebooks.utils.data import split_plain_text
from cto_notebooks.utils.rag import VectorStore, average_vector

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

if not HAS_TRAIN_DATA:
    data = load_labeled_data(label_filepath, raw_data_folder)
    tests: Dict[str, Tuple[Dict[str, Union[str, List[str]]], VectorStore]] = {}
    for record in data:
        tests[record["file_name"]] = (
            record,
            FAISS.from_documents(
                split_plain_text(
                    record["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap
                ),
                _embeddings,
            ),
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

if not HAS_TRAIN_DATA:
    queries: Dict[str, List[float]] = {
        k: average_vector(fields[k], embeddings=_embeddings, data_dict=tests.values())
        for k in questions
    }

qa_analyze_prompt = """\
### System:

Du bist ein hilfreicher Assistent. Für die folgende Aufgabe stehen dir zwischen den Tags BEGININPUT und ENDINPUT mehrere Quellen zur Verfügung. Metadaten zu den einzelnen Quellen wie Autor, URL o.ä. sind zwischen BEGINCONTEXT und ENDCONTEXT zu finden, danach folgt der Text der Quelle. Die eigentliche Aufgabe oder Frage ist zwischen BEGININSTRUCTION und ENDINSTRUCTION zu finden. Beantworte diese ausschließlich mit Informationen aus den gegebenen Quellen und gebe die Information zur genutzten Quelle unter "Quelle:" an. Sollten die Quellen keine relevanten Informationen enthalten, antworte: "Mit den gegebenen Informationen ist diese Frage nicht zu beantworten."

### User Prompt:

BEGININPUT
{context}
ENDINPUT

BEGININSTRUCTION
{question}
ENDINSTRUCTION

### Model Answer:

{target}<|im_end|>"""  # noqa: E501

# %%
# define trainingsdata functions
import json
from copy import deepcopy
from itertools import permutations

from dateparser import parse


def _format_stay_date(
    recording: str, release: str
) -> Tuple[Optional[str], Optional[str]]:
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
            msg = f"Could not parse date from {recording}!"
            raise AssertionError(msg)
        rec_date = tmp.strftime("%d.%m.%Y")
    return rec_date, rel_date


def _format_patient_date_of_birth(
    date_str: str, target_format: str = "%d.%m.%Y"
) -> Optional[str]:
    date = parse(date_str, languages=["de"], date_formats=["%Y-%m-%d"])
    return date.strftime(target_format) if date else None


def _format_patient_name(name: str) -> str:
    if name.find(", ") != -1:
        words = name.split(",")
        if len(words) == 2:
            name = words[1] + " " + words[0]
    return name.strip()


def _create_train_prompt(
    document: str,
    target: Union[str, List[str]],
    orig_json: Dict[str, Union[str, List[str]]],
) -> str:
    file_name = target.pop("file_name")

    _vectorstore: Optional[VectorStore] = None
    documents = split_plain_text(
        document, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    _vectorstore = FAISS.from_documents(documents, _embeddings)

    parameters: Dict[str, str] = {}
    prompt_list: List[(str, str)] = []

    for question in questions:
        answer = target[fields[question]]
        if not answer:
            answer = ""
        elif isinstance(answer, list):
            answer = ", ".join(answer)
        vals = orig_json[fields[question]]
        if isinstance(vals, str):
            vals = [vals]
        sorted_vecs = [
            (
                f"""\
BEGINCONTEXT
file: {file_name}
position: {doc.metadata['position']}
ENDCONTEXT
{doc.page_content}""",
                all(val in doc.page_content for val in vals),
                file_name,
                doc.metadata["position"],
            )
            for doc in _vectorstore.similarity_search_by_vector(
                queries[question], k=len(_vectorstore.index_to_docstore_id)
            )
        ]
        while not any(vec[1] for vec in sorted_vecs[:vector_count]) and sorted_vecs:
            sorted_vecs.pop(0)
        if not sorted_vecs:
            continue
        for perm_context in permutations(sorted_vecs[:vector_count]):
            match = next(con for con in perm_context if con[1])
            parameters["context"] = "\n".join(con[0] for con in perm_context)
            parameters["question"] = question
            parameters["target"] = (
                f"""\
{answer}

Quelle:
  file: {match[2]}
  position: {match[3]}
"""
                if answer
                else "Mit den gegebenen Informationen ist diese Frage nicht zu beantworten.\n"  # noqa: E501
            )
            prompt_list.append(qa_analyze_prompt.format(**parameters))
    return prompt_list


def _load_training_data(data: List) -> List[Dict]:
    label_dicts: List[Dict] = []
    for orig_json in data:
        # format names
        json_dict = deepcopy(orig_json)
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

        del json_dict["text"]

        prompt_list = _create_train_prompt(document, json_dict, orig_json)
        label_dicts.extend({"instruct": prompt} for prompt in prompt_list)

    return label_dicts


# %%
# Generate training data
from transformers import AutoTokenizer

if not HAS_TRAIN_DATA:
    training_raw_data = _load_training_data(data)
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
    n_step = 30_000

    with concurrent.futures.ProcessPoolExecutor() as executor:
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
