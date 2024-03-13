# %%
# File type defintions
from enum import Enum
from typing import Dict, Iterable, List, Optional, Protocol

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
# Define and load vector store

import json

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from cto_notebooks.utils.config import CONFIG as SETTINGS

data_dir = SETTINGS.data_dir.joinpath("grascco")
raw_data_folder = data_dir.joinpath("raw")
label_filepath = data_dir.joinpath("grascco.json")

chunk_size = 256
chunk_overlap = 25
vector_count = 3
vector_model_name = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings = HuggingFaceEmbeddings(
    model_name=vector_model_name,
    model_kwargs={"device": "cpu"},
)

fields: Dict[str, str] = {
    "PatientName": "patient_name",
    "PatientGeburtsdatum": "patient_date_of_birth",
    "BehandelnderArzt": "attending_doctor",
    "AufnahmeDatum": "recording_date",
    "EntlassDatum": "release_date",
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

# %%
# Load test kit

from typing import Tuple

with open(label_filepath, "r") as f:
    label_file = TypeAdapter(List[LabelStudioTask]).validate_python(json.load(f))

tests: Dict[str, Tuple[Dict[str, str], VectorStore]] = {}

for task in label_file:
    file_name = task.file_name
    for annotation in task.annotations:
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
            if label.name in fields:
                json_dict[fields[str(label.name)]] = value

        vectorstore = FAISS.from_texts(text_splitter.split_text(document), _embeddings)
        tests[file_name] = (json_dict, vectorstore)

# %%
# Define tests for accuracy

from collections import Counter


def collect_res(question: str, field: str) -> Dict[str, int]:
    res = {}
    for file_name, (dict, store) in tests.items():
        target_value = dict[field]
        if not target_value:
            continue
        for rank, doc in enumerate(
            store.search(question, search_type="similarity", k=vector_count), 1
        ):
            if target_value in doc.page_content:
                res[file_name] = rank
                break
        else:
            res[file_name] = 0
    return res


def benchmark_test(question: str, field: str) -> None:
    print("-" * 30)
    print(f"Frage: {question}")
    res = collect_res(question, field)
    n = len(res)
    counter = Counter(res.values())
    print(f"Sample size: {n}")
    print(f"Miss: {counter[0] / n * 100:.2f}")
    for i in range(1, vector_count + 1):
        print(f"Match Vector {i}: {counter[i] / n * 100:.2f}")
    print(f"Total hits: {(n - counter[0]) / n * 100:.2f}")


# %%
# Run tests
benchmark_test("Wie heißt der Patient?", "patient_name")
benchmark_test("Patient?", "patient_name")
benchmark_test("Patient, wh., geboren", "patient_name")

# %%
