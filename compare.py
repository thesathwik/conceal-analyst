"""Guidance diffing against last quarter, and alert de-duplication across rolling windows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from normalize import normalize_period
from schema import Alert, AlertKind, Metric, PriorGuidance

Key = tuple[str, str]  # (metric, normalized period)


@dataclass(frozen=True)
class Thresholds:
    relative_pct: float = 5.0  # CUT/RAISE when midpoint moves by more than this, relative
    margin_bps: float = 50.0  # ... or by more than this many bps for margin metrics


def key_of(m: Metric) -> Key:
    return (m.metric, normalize_period(m.period))


def is_margin(m: Metric) -> bool:
    return m.metric.endswith("_margin") or m.unit == "bps"


def load_prior(path: str | Path) -> PriorGuidance:
    return PriorGuidance.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def classify(new: Metric, prior: Metric | None, th: Thresholds) -> Alert:
    """Apply the CUT / RAISE / NEW / REITERATED rules to one guidance figure."""
    base = dict(
        metric=new.metric, period=normalize_period(new.period), unit=new.unit,
        new_low=new.value_low, new_high=new.value_high, quote=new.quote,
    )
    if prior is None or prior.unit != new.unit:
        return Alert(kind=AlertKind.NEW, **base)

    base.update(prior_low=prior.value_low, prior_high=prior.value_high)
    old_mid, new_mid = prior.midpoint, new.midpoint
    change_pct = None if old_mid == 0 else (new_mid - old_mid) / abs(old_mid) * 100

    if is_margin(new):
        delta_bps = (new_mid - old_mid) * (100 if new.unit == "percent" else 1)
        base["change_bps"] = round(delta_bps, 2)
        moved = abs(delta_bps) > th.margin_bps
        direction = delta_bps
    elif change_pct is None:
        moved, direction = new_mid != 0, new_mid
    else:
        moved, direction = abs(change_pct) > th.relative_pct, change_pct

    if change_pct is not None:
        base["change_pct"] = round(change_pct, 2)
    if not moved:
        kind = AlertKind.REITERATED
    else:
        kind = AlertKind.RAISE if direction > 0 else AlertKind.CUT
    return Alert(kind=kind, **base)


def compare(new: list[Metric], prior: list[Metric], th: Thresholds) -> list[Alert]:
    """Stateless comparison of one extraction against prior guidance. Actuals are ignored."""
    prior_by_key = {key_of(p): p for p in prior}
    return [classify(m, prior_by_key.get(key_of(m)), th) for m in new if m.type == "guidance"]


@dataclass
class GuidanceTracker:
    """Keeps the latest status per (metric, period) for one call.

    Overlapping windows re-extract the same sentence many times; an alert is
    emitted only when a key's classification or values actually change.
    """

    prior: list[Metric]
    thresholds: Thresholds = field(default_factory=Thresholds)
    latest: dict[Key, Alert] = field(default_factory=dict)

    def update(self, metrics: list[Metric]) -> list[Alert]:
        changed: list[Alert] = []
        for alert in compare(metrics, self.prior, self.thresholds):
            key = (alert.metric, alert.period)
            prev = self.latest.get(key)
            if prev is not None and _same_status(prev, alert):
                continue
            self.latest[key] = alert
            changed.append(alert)
        return changed

    def withdrawn(self) -> list[Alert]:
        """Prior guidance never repeated during the call. Call once, at the end."""
        out = []
        for p in self.prior:
            key = key_of(p)
            if key in self.latest:
                continue
            alert = Alert(
                kind=AlertKind.WITHDRAWN, metric=p.metric, period=key[1], unit=p.unit,
                prior_low=p.value_low, prior_high=p.value_high,
            )
            self.latest[key] = alert
            out.append(alert)
        return out


def _same_status(a: Alert, b: Alert) -> bool:
    return (a.kind, a.new_low, a.new_high) == (b.kind, b.new_low, b.new_high)
