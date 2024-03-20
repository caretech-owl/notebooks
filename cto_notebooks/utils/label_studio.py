from enum import Enum
from typing import Annotated, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, computed_field

T = TypeVar("T", bound=Enum)


class LabelStudioAnnotationValue(BaseModel, Generic[T]):
    end: int
    labels: List[T]
    start: int

    model_config = ConfigDict(extra="forbid")


class LabelStudioAnnotationResult(BaseModel, Generic[T]):
    from_name: str
    id: str
    origin: str
    to_name: str
    type: str
    value: LabelStudioAnnotationValue[T]

    model_config = ConfigDict(extra="forbid")


class LabelStudioAnnotation(BaseModel, Generic[T]):
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
    result: List[LabelStudioAnnotationResult[T]]
    task: int
    unique_id: str
    updated_at: str
    updated_by: int
    was_cancelled: bool

    model_config = ConfigDict(extra="forbid")


class LabelStudioTask(BaseModel, Generic[T]):
    annotations: List[LabelStudioAnnotation[T]]
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
