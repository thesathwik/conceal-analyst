"""Indian number/unit normalization and grounding checks.

The LLM is asked for values in ₹ crore, but its arithmetic is never trusted:
`reconcile()` re-parses the figures from the verbatim quote and overrides the
LLM's numbers with the parsed ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from schema import Metric

NUM = r"\d+(?:,\d+)*(?:\.\d+)?|\.\d+"
_CURRENCY = r"(?:₹|rs\.?|inr|rupees)\s*"
_SEP = r"\s*(?:to|-|–|—|till)\s*"

# Multipliers that convert a spoken amount into ₹ crore. Longest phrases first.
_INR_UNITS: list[tuple[str, float]] = [
    (r"lakh\s+crores?|lac\s+crores?|lakh-crores?", 100_000.0),
    (r"thousand\s+crores?", 1_000.0),
    (r"crores?|crs?\b", 1.0),
    (r"lakhs?|lacs?", 0.01),
    (r"billion|bn\b", 100.0),
    (r"million|mn\b", 0.1),
]
_INR_UNIT_RE = "|".join(f"(?:{u})" for u, _ in _INR_UNITS)

_INR_RE = re.compile(
    rf"(?P<cur>{_CURRENCY})?(?P<a>{NUM})(?:{_SEP}(?:{_CURRENCY})?(?P<b>{NUM}))?\s*(?P<unit>{_INR_UNIT_RE})",
    re.IGNORECASE,
)
_PCT_RE = re.compile(
    rf"(?P<a>{NUM})\s*%?(?:{_SEP}(?P<b>{NUM}))?\s*(?:%|percent\b|per\s*cent\b|pc\b)",
    re.IGNORECASE,
)
_BPS_RE = re.compile(
    rf"(?P<a>{NUM})(?:{_SEP}(?P<b>{NUM}))?\s*(?:basis\s+points?|bps\b|bp\b)",
    re.IGNORECASE,
)
_BETWEEN_RE = re.compile(rf"\bbetween\s+((?:{_CURRENCY})?(?:{NUM}))\s+and\s+", re.IGNORECASE)
_FOREIGN_BEFORE = re.compile(r"(?:\$|us\$|usd|€|eur|£)\s*$", re.IGNORECASE)
_FOREIGN_AFTER = re.compile(r"^\s*(?:us\s+)?(?:dollars?|usd|euros?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Span:
    """A figure (or range) found in text, already in the target unit."""

    low: float
    high: float

    def matches(self, low: float, high: float, rel_tol: float = 0.005) -> bool:
        return _close(self.low, low, rel_tol) and _close(self.high, high, rel_tol)


def _close(a: float, b: float, rel_tol: float) -> bool:
    if a == b:
        return True
    return abs(a - b) <= rel_tol * max(abs(a), abs(b))


def to_float(token: str) -> float:
    """'1,25,000' (Indian grouping) and '1,250.5' both parse; commas are separators only."""
    return float(token.replace(",", ""))


def _prepare(text: str) -> str:
    # "between 1,200 and 1,300 crore" -> "between 1,200 to 1,300 crore"
    return _BETWEEN_RE.sub(lambda m: f"between {m.group(1)} to ", text)


def _inr_multiplier(unit: str) -> float:
    for pattern, mult in _INR_UNITS:
        if re.fullmatch(pattern, unit.strip(), re.IGNORECASE):
            return mult
    raise ValueError(f"unknown INR unit {unit!r}")


def parse_inr_crore(text: str) -> list[Span]:
    """All rupee amounts in `text`, converted to ₹ crore.

    >>> parse_inr_crore("₹1,250 crore")[0].low
    1250.0
    >>> parse_inr_crore("12.5 billion rupees")[0].low
    1250.0
    """
    text = _prepare(text)
    spans: list[Span] = []
    for m in _INR_RE.finditer(text):
        if m.group("cur") is None and (
            _FOREIGN_BEFORE.search(text[: m.start()]) or _FOREIGN_AFTER.match(text[m.end():])
        ):
            continue  # "$1.2 billion" is not rupees
        mult = _inr_multiplier(m.group("unit"))
        a = to_float(m.group("a")) * mult
        b = to_float(m.group("b")) * mult if m.group("b") else a
        spans.append(Span(round(min(a, b), 6), round(max(a, b), 6)))
    return spans


def _parse_simple(regex: re.Pattern[str], text: str) -> list[Span]:
    spans: list[Span] = []
    for m in regex.finditer(_prepare(text)):
        a = to_float(m.group("a"))
        b = to_float(m.group("b")) if m.group("b") else a
        spans.append(Span(min(a, b), max(a, b)))
    return spans


def parse_percent(text: str) -> list[Span]:
    return _parse_simple(_PCT_RE, text)


def parse_bps(text: str) -> list[Span]:
    return _parse_simple(_BPS_RE, text)


def spans_for_unit(unit: str, text: str) -> list[Span]:
    if unit == "inr_crore":
        return parse_inr_crore(text)
    if unit == "percent":
        return parse_percent(text)
    if unit == "bps":
        return parse_bps(text)
    return []


def reconcile(metric: Metric) -> tuple[Metric | None, str]:
    """Replace the LLM's numbers with the ones parsed from its own quote.

    Returns (metric, reason). metric is None when the figure cannot be grounded:
    no figure of that unit in the quote, or several figures and none matches.
    `count` metrics are passed through (nothing to convert).
    """
    if metric.unit == "count":
        return metric, "unchecked_count"
    spans = spans_for_unit(metric.unit, metric.quote)
    if not spans:
        return None, f"no_{metric.unit}_figure_in_quote"
    for span in spans:
        if span.matches(metric.value_low, metric.value_high):
            return metric.model_copy(update={"value_low": span.low, "value_high": span.high}), "ok"
    if len(spans) == 1:
        span = spans[0]
        fixed = metric.model_copy(update={"value_low": span.low, "value_high": span.high})
        return fixed, "corrected_from_quote"
    return None, "ambiguous_figures_in_quote"


# --- periods -----------------------------------------------------------------

_FY_RE = re.compile(r"\bfy\s*'?\s*(?:(\d{2}|\d{4})\s*[-/–]\s*)?(\d{4}|\d{2})\b", re.IGNORECASE)
_Q_RE = re.compile(r"\bq\s*([1-4])(?!\d)", re.IGNORECASE)
_H_RE = re.compile(r"\b(?:h\s*([12])(?!\d)|([12])h(?![a-z]))", re.IGNORECASE)


def normalize_period(period: str) -> str:
    """'FY 2026-27', "FY'27", 'fy27', 'FY2027' -> 'FY27'; 'Q3 FY27' -> 'Q3FY27'.

    Anything unrecognised is upper-cased with whitespace removed.
    """
    p = period.strip()
    fy = _FY_RE.search(p)
    if not fy:
        return re.sub(r"\s+", "", p.upper())
    year = fy.group(2)[-2:]
    prefix = ""
    if q := _Q_RE.search(p):
        prefix = f"Q{q.group(1)}"
    elif h := _H_RE.search(p):
        prefix = f"H{h.group(1) or h.group(2)}"
    return f"{prefix}FY{year}"


# --- grounding ---------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w₹%.]+", re.UNICODE)


def normalize_for_match(text: str) -> str:
    """Lowercase, drop punctuation except decimal points, collapse whitespace."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 1,250 -> 1250
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"\.(?!\d)", " ", t)  # sentence periods, keep 12.5
    return re.sub(r"\s+", " ", t).strip()


def quote_in_window(quote: str, window: str) -> bool:
    """True if the quote is a verbatim span of the window (modulo case/punctuation/spacing)."""
    q = normalize_for_match(quote)
    return bool(q) and q in normalize_for_match(window)
