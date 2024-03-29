# %%
# Setup training parameters
import glob

from cto_notebooks.utils.config import CONFIG as SETTINGS
from cto_notebooks.utils.lora import LoraModules, LoraTrainingConfig

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
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings

# VectorStore
chunk_size = 256
chunk_overlap = 25
vector_count = 3
vector_model_name = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings = HuggingFaceEmbeddings(
    model_name=vector_model_name,
    model_kwargs={"device": "cuda"},
)

# %%
# Setup prompt


qa_analyze_prompt = """<s>Du bist ein hilfreicher Assistent. USER: \
Kontext: {context} zu
Frage: {question} in JSON-Feld answer.

Gebe nur die hilfreiche Antwort unten zurück und nichts anderes. \
Halte dich außerdem sehr kurz mit der Antwort. \
ASSISTANT:{target}</s>"""  # noqa: E501
# %%
# File type defintions
from enum import Enum
from typing import Dict, Iterable, Optional, Protocol

from langchain.docstore.document import Document
from pydantic import BaseModel, ConfigDict, TypeAdapter, computed_field


class VectorStore(Protocol):
    def merge_from(self, vector_store: "VectorStore") -> None:
        pass

    def save_local(self, path: str) -> None:
        pass

    def add_texts(
        self,
        texts: Iterable[str],
    ) -> List[str]:
        pass

    def search(self, query: str, search_type: str, k: int) -> List[Document]:
        pass


class LabelStudioLabel(Enum):
    Abteilung = "Abteilung"
    Anrede = "Anrede"
    AufnahmeDatum = "AufnahmeDatum"
    BehandelnderArzt = "BehandelnderArzt"
    Einrichtung = "Einrichtung"
    EntlassDatum = "EntlassDatum"
    Hausarzt = "Hausarzt"
    PatientGeburtsdatum = "PatientGeburtsdatum"
    PatientName = "PatientName"


class LabelStudioAnnotationValue(BaseModel):
    end: int
    labels: List[LabelStudioLabel]
    start: int

    model_config = ConfigDict(extra="forbid")


class LabelStudioAnnotationResult(BaseModel):
    from_name: str
    id: str
    origin: str
    to_name: str
    type: str
    value: LabelStudioAnnotationValue

    model_config = ConfigDict(extra="forbid")


class LabelStudioAnnotation(BaseModel):
    completed_by: int
    created_at: str
    draft_created_at: Optional[str]
    ground_truth: bool
    id: int
    import_id: Optional[str]
    last_action: Optional[str]
    last_created_by: Optional[int]
    lead_time: float
    parent_annotation: Optional[str]
    parent_prediction: Optional[str]
    prediction: Dict[str, str]
    project: int
    result_count: int
    result: List[LabelStudioAnnotationResult]
    task: int
    unique_id: str
    updated_at: str
    updated_by: int
    was_cancelled: bool

    model_config = ConfigDict(extra="forbid")


class LabelStudioTask(BaseModel):
    annotations: List[LabelStudioAnnotation]
    cancelled_annotations: int
    comment_authors: List[str]
    comment_count: int
    created_at: str
    data: Optional[Dict[str, str]]
    drafts: List[str]
    file_upload: str
    id: int
    inner_id: int
    last_comment_updated_at: Optional[str]
    meta: Optional[Dict[str, str]]
    predictions: List[str]
    project: int
    total_annotations: int
    total_predictions: int
    unresolved_comment_count: int
    updated_at: str
    updated_by: int

    @computed_field  # type: ignore[misc]
    @property
    def file_name(self) -> str:
        return self.file_upload.split("-", 1)[-1]

    model_config = ConfigDict(extra="forbid")


# %%
# define LabelStudio helper functions

import json
from itertools import permutations

from datasets import Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def load_label_studio_tasks(file_path: str) -> List[LabelStudioTask]:
    with open(file_path, "r") as f:
        obj = json.load(f)

    tasks = TypeAdapter(List[LabelStudioTask]).validate_python(obj)
    return tasks

# %% define trainingsdata functions
import re
from datetime import datetime


def _convert_to_datetime(date_string: str) -> datetime:
    if re.match(r"\d{1,2}\.\d{1,2}\.\d{4}$", date_string):
        return datetime.strptime(date_string, "%d.%m.%Y")
    elif re.match(r"\d{1,2}\.\d{1,2}\.\d{2}$", date_string):
        return datetime.strptime(date_string, "%d.%m.%y")
    elif re.match(r"\d{1,2}\-\d{1,2}\-\d{4}$", date_string):
        return datetime.strptime(date_string, "%d-%m-%Y")
    elif re.match(r"\d{1,2}\-\d{1,2}\-\d{2}$", date_string):
        return datetime.strptime(date_string, "%d-%m-%y")
    elif re.match(r"\d{1,2}/\d{1,2}/\d{4}$", date_string):
        return datetime.strptime(date_string, "%d/%m/%Y")
    elif re.match(r"\d{1,2}/\d{1,2}/\d{2}$", date_string):
        return datetime.strptime(date_string, "%d/%m/%y")
    elif re.match(r"\d{4}\-\d{2}\-\d{2}$", date_string):
        return datetime.strptime(date_string, "%Y-%m-%d")

def _replace_month_names(date : str) -> str:
    month_dict = {"Januar": "01",
                "Februar": "02",
                "März": "03",
                "April": "04",
                "Mai": "05",
                "Juni": "06",
                "Juli": "07",
                "August": "08",
                "September": "09",
                "Oktober": "10",
                "November": "11",
                "Dezember": "12"}

    for month_name, month_number in month_dict.items():
        if any(re.finditer(month_name, date)):
            date = date.replace(" ", "")

            date = date.replace("." + month_name + ".", "." + month_number + ".")
            date = date.replace(month_name + ".", "." + month_number + ".")
            date = date.replace("." + month_name, "." + month_number + ".")
            date = date.replace(month_name, "." + month_number + ".")
    return date

def _format_stay_date(
        recording_date_str: str, release_date_str: str, target_format: str = "%d.%m.%Y"
        ) -> tuple[str, str]:
    formatted_recording_date : str = recording_date_str
    formatted_release_date : str = release_date_str
    recording_date : datetime = None
    release_date : datetime = None

    formatted_recording_date = _replace_month_names(
        formatted_recording_date
        ).replace(" ", "")
    formatted_release_date = _replace_month_names(
        formatted_release_date
        ).replace(" ", "")

    if not formatted_recording_date.isspace() and  formatted_recording_date != "":
        recording_date = _convert_to_datetime(formatted_recording_date)
        if recording_date is not None:
            formatted_recording_date = recording_date.strftime(target_format)

    if not formatted_release_date.isspace() and  formatted_release_date != "":
        release_date = _convert_to_datetime(formatted_release_date)
        if release_date is not None:
            formatted_release_date = release_date.strftime(target_format)

    if recording_date is None and release_date is not None:
        if re.match(r"\d{1,2}\.\d{1,2}\.", formatted_recording_date):
            formatted_recording_date = formatted_recording_date + str(release_date.year)
            recording_date = _convert_to_datetime(formatted_recording_date)
            if recording_date is not None and recording_date <= release_date:
                    formatted_recording_date = recording_date.strftime(target_format)
        elif re.match(r"\d{1,2}\.", formatted_recording_date):
            formatted_recording_date = (formatted_recording_date
                                        + str(release_date.month)
                                        + "." + str(release_date.year))
            recording_date = _convert_to_datetime(formatted_recording_date)
            if recording_date is not None and recording_date <= release_date:
                    formatted_recording_date = recording_date.strftime(target_format)
    return (formatted_recording_date, formatted_release_date)

def _format_patient_date_of_birth(
        patient_date_of_birth_str: str, target_format: str = "%d.%m.%Y"
        ) -> str:
    formatted_patient_date_of_birth = _replace_month_names(
        patient_date_of_birth_str
        ).replace(" ", "")

    if (not formatted_patient_date_of_birth.isspace()
        and  formatted_patient_date_of_birth != ""):
        patient_date_of_birth = _convert_to_datetime(formatted_patient_date_of_birth)
        if patient_date_of_birth is not None:
            formatted_patient_date_of_birth = patient_date_of_birth.strftime(
                target_format
                )

    return formatted_patient_date_of_birth

def _format_patient_name(name: str) -> str:
    if name.find(", ") != -1:
        words = name.split(",")
        if len(words) == 2:
            name = words[1] + " " + words[0]
    return name.strip()

def _replace_with_german_filenames(file_name: str) -> str:
    german_file_names: Dict[str, str] = {
        "Waldenstrom.txt": "Waldenström.txt",
        "Stolzl.txt": "Stölzl.txt",
    }

    if file_name in german_file_names:
        return german_file_names[file_name]
    return file_name


def _create_train_prompt(document: str, target_dict: str) -> str:
    _vectorstore: Optional[VectorStore] = None
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = text_splitter.split_text(document)
    _vectorstore = FAISS.from_texts(texts, _embeddings)

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
        "Wie heißt der Arzt?": "attending_doctor",
        "Wann wurde der Patient bei uns aufgenommen?": "recording_date",
        "Wann wurde der Patient bei uns entlassen?": "release_date",
    }

    parameters: Dict[str, str] = {}
    prompt_list: List[(str, str)] = []

    for question in questions:
        parameters["target"] = ("{question: "
                                + str(question) + ", answer: "
                                + str(target_dict[fields[question]]) + "}")

        context = [
            doc.page_content
            for doc in _vectorstore.search(
                question, search_type="similarity", k=vector_count
            )
        ]

        for perm_context in permutations(context):
            parameters["context"] = " ".join(perm_context)
            parameters["question"] = question
            prompt_list.append(
                (parameters["target"], qa_analyze_prompt.format(**parameters))
                )
    return prompt_list


def _load_training_data() -> List[Dict]:
    label_file = load_label_studio_tasks(label_filepath)
    label_dicts: List[Dict] = []
    fields: Dict[str, str] = {
        "PatientName": "patient_name",
        "PatientGeburtsdatum": "patient_date_of_birth",
        "BehandelnderArzt": "attending_doctor",
        "AufnahmeDatum": "recording_date",
        "EntlassDatum": "release_date",
    }

    for task in label_file:
        file_name = _replace_with_german_filenames(task.file_name)

        for annotation in task.annotations:
            if file_name not in exclude_from_training:
                with open(
                    raw_data_folder.joinpath(file_name), encoding="utf-8-sig"
                ) as document_file:
                    document = document_file.read()

                json_dict: Dict[str, str] = {
                    "patient_name": "",
                    "patient_date_of_birth": "",
                    "attending_doctor": "",
                    "recording_date": "",
                    "release_date": "",
                }
                for res in annotation.result:
                    label = res.value.labels[0]
                    start = res.value.start
                    end = res.value.end
                    value = document[start:end]
                    if str(label.name) in fields:
                        if (json_dict[fields[str(label.name)]] != ""
                        and fields[str(label.name)] == "attending_doctor"):
                            json_dict[fields[str(label.name)]] = json_dict[
                                fields[str(label.name)]
                                ] + ", " + value
                        else:
                            json_dict[fields[str(label.name)]] = value

                # format names
                json_dict["patient_name"] = json_dict[
                    "patient_name"
                    ].replace("\n", " ")
                json_dict["attending_doctor"] = json_dict[
                    "attending_doctor"
                    ].replace("\n", " ")

                json_dict["patient_name"] = _format_patient_name(
                    json_dict["patient_name"]
                    )

                # format dates
                json_dict["patient_date_of_birth"] = _format_patient_date_of_birth(
                    json_dict["patient_date_of_birth"]
                    )

                formatted_dates = _format_stay_date(
                    json_dict["recording_date"],
                    json_dict["release_date"]
                    )

                json_dict["recording_date"] = formatted_dates[0]
                json_dict["release_date"] = formatted_dates[1]

                #target = json.dumps(json_dict)

                prompt_list = _create_train_prompt(document, json_dict)

                for prompt in prompt_list:
                    label_dict: Dict[str, str] = {}
                    label_dict["target"] = prompt[0]
                    label_dict["instruct"] = prompt[1]

                    label_dicts.append(label_dict)

    return label_dicts


# %%
# Generate training data
from transformers import AutoTokenizer

if not HAS_TRAIN_DATA:
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
