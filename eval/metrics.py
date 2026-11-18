"""Pure scoring functions for validation and benchmarks (unit-tested; no I/O)."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from normalize import normalize_period
from schema import Metric

# --- ASR ---------------------------------------------------------------------

_SPEAKER_RE = re.compile(r"^\s*[A-Z][\w.' -]{0,60}:\s*", re.MULTILINE)
_REPLACEMENTS = [
    (r"(?<=\d),(?=\d)", ""),          # 1,25,000 -> 125000
    (r"%", " percent "),
    (r"\bper\s+cent\b", "percent"),
    (r"₹|\brs\.?(?=\s|\d)|\binr\b", " rupees "),
    (r"\bcrores\b", "crore"),
    (r"\blakhs\b|\blacs?\b", "lakh"),
    (r"\bbps\b|\bbasis points?\b", "basis points"),
    (r"\be\.?b\.?i\.?t\.?d\.?a\b", "ebitda"),
    (r"\bfy\s*'?\s*(\d{2})\b", r"fy\1"),
    (r"(\d)\s*-\s*(\d)", r"\1 to \2"),
]


def strip_speaker_labels(text: str) -> str:
    """Exchange-filed transcripts prefix turns with 'Name:'; ASR output has none."""
    return _SPEAKER_RE.sub("", text)


def normalize_for_wer(text: str) -> str:
    t = text.lower()
    for pattern, repl in _REPLACEMENTS:
        t = re.sub(pattern, repl, t)
    t = re.sub(r"[^\w\s.]", " ", t)
    t = re.sub(r"\.(?!\d)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def term_error_rate(reference: str, hypothesis: str, terms: Iterable[str]) -> tuple[float | None, int]:
    """Share of finance-term occurrences in the reference that the hypothesis misses.

    Counts are compared per term (order-free), so it measures vocabulary recognition, not alignment.
    Inputs should already be normalized with normalize_for_wer. Returns (rate, reference occurrences).
    """
    missed = total = 0
    for term in terms:
        t = normalize_for_wer(term)
        if not t:
            continue
        pat = re.compile(rf"\b{re.escape(t)}\b")
        ref_n, hyp_n = len(pat.findall(reference)), len(pat.findall(hypothesis))
        total += ref_n
        missed += max(ref_n - hyp_n, 0)
    return (missed / total if total else None), total


# --- numeric extraction --------------------------------------------------------


@dataclass(frozen=True)
class Figure:
    metric: str
    period: str
    low: float
    high: float
    unit: str

    @classmethod
    def of(cls, m: Metric) -> "Figure":
        return cls(m.metric, normalize_period(m.period), round(m.value_low, 6), round(m.value_high, 6), m.unit)


def _value_match(a: float, b: float, unit: str) -> bool:
    if unit == "bps":
        return a == b
    if a == b:
        return True
    return abs(a - b) <= 0.005 * max(abs(a), abs(b))


def figures_match(x: Figure, gold: Figure) -> bool:
    return (
        (x.metric, x.period) == (gold.metric, gold.period)
        and _value_match(x.low, gold.low, gold.unit)
        and _value_match(x.high, gold.high, gold.unit)
    )


@dataclass
class NumericScore:
    n_gold: int
    n_extracted: int
    recall: float | None
    precision: float | None
    wrong_inr: list[tuple[Figure, Figure]]  # (extracted, gold) same metric+period, ₹ value wrong


def score_numbers(extracted: Iterable[Figure], gold: Sequence[Figure]) -> NumericScore:
    ext = list(dict.fromkeys(extracted))  # unique, order kept
    found = {g for g in gold if any(figures_match(e, g) for e in ext)}
    correct = [e for e in ext if any(figures_match(e, g) for g in gold)]
    wrong_inr = [
        (e, g)
        for e in ext
        if e.unit == "inr_crore" and e not in correct
        for g in gold
        if g.unit == "inr_crore" and (e.metric, e.period) == (g.metric, g.period)
    ]
    return NumericScore(
        n_gold=len(gold),
        n_extracted=len(ext),
        recall=len(found) / len(gold) if gold else None,
        precision=len(correct) / len(ext) if ext else None,
        wrong_inr=wrong_inr,
    )


# --- latency -------------------------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (numpy's default), p in [0, 100]."""
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    vals = [v for v in values if v is not None]
    return {
        "count": len(vals),
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
        "mean": sum(vals) / len(vals) if vals else None,
    }
