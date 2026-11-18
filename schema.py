"""Pydantic models for every piece of model I/O: extraction output, prior guidance, alerts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

MetricName = Literal[
    "revenue_growth", "revenue", "ebitda_margin", "pat", "capex", "order_book", "net_debt", "other"
]
Unit = Literal["percent", "inr_crore", "bps", "count"]
MetricType = Literal["guidance", "actual"]


class Metric(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric: MetricName
    value_low: float
    value_high: float
    unit: Unit
    period: str = Field(min_length=1)
    type: MetricType
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def _order_range(self) -> "Metric":
        if self.value_low > self.value_high:
            self.value_low, self.value_high = self.value_high, self.value_low
        return self

    @property
    def midpoint(self) -> float:
        return (self.value_low + self.value_high) / 2


class ToneFlag(BaseModel):
    model_config = ConfigDict(extra="ignore")

    flag: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class Extraction(BaseModel):
    metrics: list[Metric] = Field(default_factory=list)
    tone_flags: list[ToneFlag] = Field(default_factory=list)


class ParsedExtraction(BaseModel):
    """Extraction plus the items that failed validation (kept for logging, never used)."""

    extraction: Extraction
    rejected: list[dict[str, Any]] = Field(default_factory=list)


def parse_extraction(data: Any) -> ParsedExtraction:
    """Validate an LLM JSON object item by item.

    One malformed metric must not throw away the rest of the window, so items are
    validated individually and failures are returned in `rejected`.
    Raises ValueError if the top-level shape is wrong (that triggers the repair prompt).
    """
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    raw_metrics = data.get("metrics", [])
    raw_flags = data.get("tone_flags", [])
    if not isinstance(raw_metrics, list) or not isinstance(raw_flags, list):
        raise ValueError("'metrics' and 'tone_flags' must be arrays")

    metrics: list[Metric] = []
    flags: list[ToneFlag] = []
    rejected: list[dict[str, Any]] = []
    for item in raw_metrics:
        try:
            metrics.append(Metric.model_validate(item))
        except ValidationError as exc:
            rejected.append({"item": item, "error": exc.errors(include_url=False)[0]["msg"]})
    for item in raw_flags:
        try:
            flags.append(ToneFlag.model_validate(item))
        except ValidationError as exc:
            rejected.append({"item": item, "error": exc.errors(include_url=False)[0]["msg"]})
    return ParsedExtraction(extraction=Extraction(metrics=metrics, tone_flags=flags), rejected=rejected)


class PriorGuidance(BaseModel):
    """Shape of data/last_quarter.json."""

    model_config = ConfigDict(extra="ignore")

    company: str = ""
    quarter: str = ""
    metrics: list[Metric] = Field(default_factory=list)

    @model_validator(mode="after")
    def _guidance_only(self) -> "PriorGuidance":
        bad = [m.metric for m in self.metrics if m.type != "guidance"]
        if bad:
            raise ValueError(f"last_quarter.json must contain guidance only; got actuals for {bad}")
        return self


class AlertKind(str, Enum):
    CUT = "CUT"
    RAISE = "RAISE"
    NEW = "NEW"
    REITERATED = "REITERATED"
    WITHDRAWN = "WITHDRAWN"

    @property
    def notifies(self) -> bool:
        """REITERATED is shown in the UI but is not an alert."""
        return self is not AlertKind.REITERATED


class Alert(BaseModel):
    kind: AlertKind
    metric: MetricName
    period: str
    unit: Unit
    new_low: float | None = None
    new_high: float | None = None
    prior_low: float | None = None
    prior_high: float | None = None
    change_pct: float | None = None  # relative change of midpoint, percent
    change_bps: float | None = None  # absolute change for margin-type metrics
    quote: str = ""
