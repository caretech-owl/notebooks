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
        "tokenizer": "LeoLM/leo-hessianai-7b-chat",
        "config": {"pretrained_model_name_or_path": "LeoLM/leo-hessianai-7b-chat"},
    },
    batch_size=4,
    micro_batch_size=1,
    cutoff_len=2200,
    stop_at_loss=-1,
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
vector_count = 2
vector_model_name = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings = HuggingFaceEmbeddings(
    model_name=vector_model_name,
    model_kwargs={"device": "cuda"},
)

# %%
# Setup prompt

qa_analyze_prompt = """<|im_start|>user
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

Gebe nur die hilfreichen Antworten unten zurück und nichts anderes. Halte dich außerdem sehr kurz mit der Antwort.
Hilfreiche Antwort:<|im_end|>
<|im_start|>assistant
{target}<|im_end|>"""  # noqa: E501
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


def _replace_with_german_filenames(file_name: str) -> str:
    german_file_names: Dict[str, str] = {
        "Waldenstrom.txt": "Waldenström.txt",
        "Stolzl.txt": "Stölzl.txt",
    }

    if file_name in german_file_names:
        return german_file_names[file_name]
    return file_name


def _create_train_prompt(document: str, target: str) -> str:
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

    questions_dict: Dict[str, str] = {}
    parameters: Dict[str, str] = {}
    prompt_list: List[str] = []

    parameters["target"] = target

    for question in questions:
        questions_dict[question] = [
            doc.page_content
            for doc in _vectorstore.search(question, search_type="similarity", k=2)
        ]

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
                    raw_data_folder.joinpath(file_name), encoding="utf-8"
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
                        json_dict[fields[str(label.name)]] = value

                target = json.dumps(json_dict)
                prompt_list = _create_train_prompt(document, target)

                for prompt in prompt_list:
                    label_dict: Dict[str, str] = {}
                    label_dict["target"] = target
                    label_dict["instruct"] = prompt

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

from cto_notebooks.utils.data import tokenize

torch.multiprocessing.set_sharing_strategy("file_system")


def tokenizer_worker(args: List) -> List[Dict[str, List]]:
    samples, start = args
    tokenizer = AutoTokenizer.from_pretrained(
        **config.model["config"], trust_remote_code=False
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    tokenized = [
        tokenize(s["instruct"], tokenizer, config.cutoff_len)
        for s in samples
        if len(s["instruct"]) <= config.cutoff_len
    ]
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
tokenizer.padding_side = "left"

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
## Step 6 - Run Trainer
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
