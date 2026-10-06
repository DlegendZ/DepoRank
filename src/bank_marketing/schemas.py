"""Shared domain and dataset contracts."""

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DomainError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details
        self.error_id = uuid4().hex
CORE_FEATURES = ("age", "job", "marital", "education", "default", "housing", "loan", "pdays", "previous", "poutcome")
CONTEXT_FEATURES = CORE_FEATURES + ("contact", "month", "day_of_week", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")
TRAINING_COLUMNS = ("age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y")
CATEGORIES = {
    "job": ("admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"),
    "marital": ("divorced", "married", "single", "unknown"),
    "education": ("basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"),
    "default": ("no", "yes", "unknown"), "housing": ("no", "yes", "unknown"), "loan": ("no", "yes", "unknown"),
    "contact": ("cellular", "telephone"),
    "month": ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
    "day_of_week": ("mon", "tue", "wed", "thu", "fri"),
    "poutcome": ("failure", "nonexistent", "success"), "y": ("no", "yes"),
}
INTEGER_COLUMNS = ("age", "duration", "campaign", "pdays", "previous")
NUMERIC_COLUMNS = INTEGER_COLUMNS + ("emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")
DATASET_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
DICTIONARY_SHA256 = "10045462a4c026f858dffa159370c4270510230c1fc63c6c91ca51bba4b3df26"
OFFICIAL_URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"
SPLIT_RANGES = {"train": [0, 24712], "calibration": [24712, 28831], "selection": [28831, 32950], "test": [32950, 41188]}


class ScoreRow(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    input_row_id: int = Field(ge=0)
    record_id: str = Field(min_length=1)
    p_yes: float = Field(ge=0, le=1)
    predicted_label: Literal["yes", "no"]
    warnings: list[dict] = Field(default_factory=list)


class ScoreResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    batch_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    feature_set: Literal["core_v1", "context_v1"]
    threshold: float = Field(ge=0, le=1)
    rows: list[ScoreRow]
    warnings: list[dict] = Field(default_factory=list)
    model_hash: str | None = None
    created_at_utc: str | None = None

    @model_validator(mode="after")
    def unique_row_ids(self):
        ids = [row.input_row_id for row in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError("input_row_id must be unique")
        return self


class RankingRow(ScoreRow):
    rank: int = Field(ge=1)
    selected_for_contact: bool


class RankingResult(ScoreResult):
    rows: list[RankingRow]
    capacity_k: int = Field(ge=0)
    call_fraction: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def consistent_ranking(self):
        n = len(self.rows)
        if self.capacity_k > n:
            raise ValueError("capacity_k exceeds row count")
        expected_fraction = self.capacity_k / n if n else 0.0
        if abs(self.call_fraction - expected_fraction) > 1e-12:
            raise ValueError("call_fraction must match capacity_k / row count")
        if [row.rank for row in self.rows] != list(range(1, n + 1)):
            raise ValueError("Ranks must be consecutive in output order")
        keys = [(-row.p_yes, row.input_row_id) for row in self.rows]
        if keys != sorted(keys):
            raise ValueError("Rows must use descending probability and ascending row ID ties")
        if any(row.selected_for_contact != (row.rank <= self.capacity_k) for row in self.rows):
            raise ValueError("Selection flags must select exactly top capacity_k rows")
        return self

