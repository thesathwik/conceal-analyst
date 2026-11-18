"""One extraction pass over a transcript window: prompt -> JSON -> validate -> ground -> normalize.

Invalid JSON is retried once with a repair prompt; after that the window is
dropped and the failure is logged by the caller.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clients import LLMClient, LLMResult
from normalize import normalize_period, quote_in_window, reconcile
from schema import Extraction, Metric, ToneFlag, parse_extraction

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

REPAIR_PROMPT = (
    "Your previous reply was not valid for the required schema.\n"
    "Error: {error}\n"
    "Reply again with ONLY the corrected JSON object, using the same schema and the same transcript."
)


@dataclass
class ExtractionOutcome:
    extraction: Extraction | None  # None = dropped after a failed repair
    calls: list[LLMResult] = field(default_factory=list)
    repaired: bool = False
    error: str | None = None
    rejected: list[dict[str, Any]] = field(default_factory=list)  # schema / grounding / unit failures

    @property
    def llm_latency_s(self) -> float:
        return sum(c.total_s for c in self.calls)


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def build_messages(system_prompt: str, window_text: str) -> list[dict[str, str]]:
    user = f"Transcript window:\n<<<\n{window_text}\n>>>\n\nReturn the JSON object now."
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def parse_json_object(text: str) -> Any:
    """Parse the model's reply; tolerate markdown fences and leading/trailing chatter."""
    t = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(t[start : end + 1])


def ground_and_normalize(extraction: Extraction, window_text: str) -> tuple[Extraction, list[dict[str, Any]]]:
    """Drop items whose quote is not in the window; re-derive numbers from quotes; canonical periods."""
    kept: list[Metric] = []
    rejected: list[dict[str, Any]] = []
    for m in extraction.metrics:
        if not quote_in_window(m.quote, window_text):
            rejected.append({"item": m.model_dump(), "error": "quote_not_in_window"})
            continue
        fixed, reason = reconcile(m)
        if fixed is None:
            rejected.append({"item": m.model_dump(), "error": reason})
            continue
        kept.append(fixed.model_copy(update={"period": normalize_period(fixed.period)}))
    flags: list[ToneFlag] = [f for f in extraction.tone_flags if quote_in_window(f.quote, window_text)]
    return Extraction(metrics=kept, tone_flags=flags), rejected


async def extract_window(llm: LLMClient, system_prompt: str, window_text: str) -> ExtractionOutcome:
    messages = build_messages(system_prompt, window_text)
    outcome = ExtractionOutcome(extraction=None)
    for attempt in range(2):
        result = await llm.chat(messages)
        outcome.calls.append(result)
        try:
            parsed = parse_extraction(parse_json_object(result.text))
        except (ValueError, json.JSONDecodeError) as exc:  # JSONDecodeError is a ValueError; explicit for clarity
            outcome.error = f"{type(exc).__name__}: {exc}"
            if attempt == 0:
                outcome.repaired = True
                messages = messages + [
                    {"role": "assistant", "content": result.text},
                    {"role": "user", "content": REPAIR_PROMPT.format(error=outcome.error)},
                ]
            continue
        grounded, ungrounded = ground_and_normalize(parsed.extraction, window_text)
        outcome.extraction = grounded
        outcome.rejected = parsed.rejected + ungrounded
        outcome.error = None
        return outcome
    return outcome
