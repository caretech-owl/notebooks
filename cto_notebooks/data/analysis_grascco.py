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
chunk_overlap = 26
vector_count = 3
vector_model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
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
            if label.name in fields:
                json_dict[fields[str(label.name)]] = value
        # print(json_dict["patient_name"])
        vectorstore = FAISS.from_texts(text_splitter.split_text(document), _embeddings)
        tests[file_name] = (json_dict, vectorstore)

# %%
# Define tests for accuracy

from collections import Counter
from typing import Dict, List, Union


def collect_res(
    question: Union[str, List[float]], field: str, k: int
) -> Dict[str, int]:
    res = {}
    for file_name, (dict, store) in tests.items():
        target_value = dict[field]
        if not target_value:
            continue
        for rank, doc in enumerate(
            store.search(question, search_type="similarity", k=k)
            if isinstance(question, str)
            else store.similarity_search_by_vector(
                question, search_type="similarity", k=k
            ),
            1,
        ):
            if target_value in doc.page_content:
                res[file_name] = rank
                break
        else:
            # print(file_name)
            res[file_name] = 0
    return res


def benchmark_test(question: Union[str, List[float]], field: str, k: int = 3) -> None:
    print("-" * 30)
    if isinstance(question, str):
        print(f"Frage: {question}")
    else:
        print("Frage: Vector")
    res = collect_res(question, field, k)
    n = len(res)
    counter = Counter(res.values())
    misses = counter[0] / n * 100
    print(f"Sample size: {n}")
    print(f"Miss: {misses:.2f}")
    for i in range(1, k + 1):
        print(f"Match Vector {i}: {counter[i] / n * 100:.2f}")
    print(f"Total hits: {100 - misses:.2f}")


# %%
# Run tests
benchmark_test("Wie heißt der Patient?", "patient_name")
benchmark_test("Patient?", "patient_name")
benchmark_test("Patient, wh., geboren", "patient_name")


# %%
def average_vector(field: str) -> List[float]:
    vecs = []
    for dict, store in tests.values():
        for vec in store.search(
            f"wir berichten über unseren gemeinsamen Patienten {dict[field]}",
            search_type="similarity",
            k=10,
        ):
            if dict[field] not in vec.page_content:
                continue
            # print(vec.page_content)
            vecs.append(_embeddings.embed_query(vec.page_content))
            break
        # else:
        #     msg = f"Field value: {dict[field]} not in document {file_name}."
        #     raise ValueError(msg)
        _embeddings.embed_query(vec.page_content)
    return numpy.mean(vecs, axis=0)


# %%
# Calcalate and test average vector for patient_name
import numpy

m_vec = average_vector("patient_name")
benchmark_test(m_vec, "patient_name", 3)

# %%
# Calculate and test average vector for attending_doctor
m_vec = average_vector("attending_doctor")
benchmark_test(m_vec, "attending_doctor", 3)

# %%
# Define batch benchmark ranking tests
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    question: str
    sample_size: float
    miss: float
    vector_matches: List[float]

    @property
    def total_hits(self) -> float:
        return 100 - self.miss


def benchmark_tests(
    question_list: List[str], field: str, num_show_first: int = 0, k: int = 3
) -> None:
    benchmark_results: List[BenchmarkResult] = []
    for question in question_list:
        res = collect_res(question, field, k)
        counter = Counter(res.values())
        size = len(res)
        benchmark_re = BenchmarkResult(
            question=question,
            sample_size=size,
            miss=counter[0] / size * 100,
            vector_matches=[counter[i] / size * 100 for i in range(1, k + 1)],
        )
        benchmark_results.append(benchmark_re)

    # sort
    benchmark_results = sorted(
        benchmark_results, key=lambda res: res.total_hits, reverse=True
    )

    # print
    if num_show_first == 0 or num_show_first > len(benchmark_results):
        num_show_first = len(benchmark_results)

    for i in range(0, num_show_first):
        bench_re = benchmark_results[i]
        print("-" * 30)
        print(f"Frage: {bench_re.question}")

        print(f"Sample size: {bench_re.sample_size}")
        print(f"Miss: {bench_re.miss:.2f}")
        for i in range(0, len(bench_re.vector_matches)):
            print(f"Match Vector {i + 1}: {bench_re.vector_matches[i]:.2f}")
        print(f"Total hits: {bench_re.total_hits:.2f}")
    print("-" * 30)


# %%
# patient_name
questions: List[str] = [
    "Wie heißt der Patient?",
    "Patient?",
    "Patientname?",
    "Patient of patient?",
    "Patientennamen?",
    "Patient, wh., geboren",
    "Patient, wh., geboren, ?",
    "wir berichten über unseren Patient, Btr.",
    "wir berichten über unseren Patient",
    "wir berichten über",
    "wir berichten über unseren Patient oder Btr.",
    "wir berichten über unseren Patient oder Btr. oder Patient, * 0.00.0000,",
    "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren",
    "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,",  # noqa: E501
    "Patient, * 0.00.0000,",
]
benchmark_tests(questions, "patient_name", 4)

# %%
# Test questions for patient_date_of_birth
questions: List[str] = [
    "Patient, wh., geboren",
    "Patient, wh., geboren, ?",
    "Wann ist der Patient geboren?",
    "Patient Geburtstag",
    "Patient Geburtstag?",
    "Patient, born",
    "Patient, born?",
    "wir berichten über unseren Patient oder Btr.",
    "wir berichten über unseren Patient oder Btr. oder Patient, * 0.00.0000,",
    "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren",
    "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,",  # noqa: E501
]
benchmark_tests(questions, "patient_date_of_birth", 4)


# %%
# Test questions for recording_date
questions: List[str] = [
    "Wann ist der Patient gegangen?",
    "Wann wurde der Patient bei uns entlassen?",
    "wir berichten über unseren Patient oder Btr.",
    "wir berichten über unseren Patient oder Btr. oder Patient, * 0.00.0000,",
    "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren",
    "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,",  # noqa: E501
]
benchmark_tests(questions, "recording_date", 4)

# %%
# Test questions for release_date
questions: List[str] = [
    "Wann ist der Patient gekommen?",
    "Wann wurde der Patient bei uns aufgenommen?",
    "wir berichten über unseren Patient oder Btr.",
    "wir berichten über unseren Patient oder Btr. oder Patient, * 0.00.0000,",
    "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren",
    "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,",  # noqa: E501
]
benchmark_tests(questions, "release_date", 4)

# %%
# Test questions for attending_doctor
questions: List[str] = [
    "Wie heißt der behandelnde Arzt?",
    "Arzt",
    "Was ist der Name des behandelnden Arztes?",
    "Grüßen , Prof, Dr",
    "Grüßen",
    "Grüße , Prof, Dr",
    "Mit freundlichen kollegialen Grüßen, Prof, Dr",
    "Mit freundlichen kollegialen Grüßen",
    "Mit freundlichen Grüßen, Prof, Dr",
    "Mit freundlichen Grüßen",
]
benchmark_tests(questions, "attending_doctor", 4)
