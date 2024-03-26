import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Union

from pydantic import TypeAdapter

from cto_notebooks.utils.label_studio import LabelStudioTask


class GrasccoLabel(Enum):
    Abteilung = "Abteilung"
    Anrede = "Anrede"
    AufnahmeDatum = "AufnahmeDatum"
    BehandelnderArzt = "BehandelnderArzt"
    Einrichtung = "Einrichtung"
    EntlassDatum = "EntlassDatum"
    Hausarzt = "Hausarzt"
    PatientGeburtsdatum = "PatientGeburtsdatum"
    PatientName = "PatientName"


fields = {
    GrasccoLabel.PatientName.value: "patient_name",
    GrasccoLabel.PatientGeburtsdatum.value: "patient_date_of_birth",
    GrasccoLabel.BehandelnderArzt.value: "attending_doctors",
    GrasccoLabel.AufnahmeDatum.value: "recording_date",
    GrasccoLabel.EntlassDatum.value: "release_date",
}


class GrasccoTask(LabelStudioTask[GrasccoLabel]):
    pass


def load_labeled_data(
    label_filepath: Path, raw_data_folder: Path
) -> List[Dict[str, Union[str, List[str]]]]:
    with open(label_filepath, "r") as f:
        label_file = TypeAdapter(List[GrasccoTask]).validate_python(json.load(f))

    data = []
    for task in label_file:
        file_name = task.file_name
        for annotation in task.annotations:
            with open(
                raw_data_folder.joinpath(file_name), encoding="utf-8-sig"
            ) as document_file:
                document = document_file.read()

            json_dict: Dict[str, str] = {
                "file_name": file_name,
                "patient_name": "",
                "patient_date_of_birth": "",
                "attending_doctors": [],
                "recording_date": "",
                "release_date": "",
                "text": document,
            }

            for res in annotation.result:
                label = res.value.labels[0]
                value = document[res.value.start : res.value.end]
                label_name = str(label.name)
                if label_name in fields:
                    field_name = fields[label_name]
                    if isinstance(json_dict[field_name], list):
                        json_dict[field_name].append(value)
                    else:
                        json_dict[field_name] = value
        data.append(json_dict)
    return data
