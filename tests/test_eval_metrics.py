import pytest

from eval.bench import summarize_level
from eval.gpu import extract_utilisation
from eval.metrics import (
    Figure, normalize_for_wer, percentile, score_numbers, strip_speaker_labels, term_error_rate,
)
from eval.validate import apply_gate


def test_wer_normalization_unifies_numbers_and_units():
    assert normalize_for_wer("EBITDA of ₹1,250 Crores, so 16-17%.") == \
        normalize_for_wer("ebitda of rupees 1250 crore so 16 to 17 percent")


def test_strip_speaker_labels():
    ref = "Moderator: Welcome everyone.\nRahul Sharma: Thank you. Margins: strong."
    assert strip_speaker_labels(ref).split("\n") == ["Welcome everyone.", "Thank you. Margins: strong."]


def test_term_error_rate():
    rate, n = term_error_rate("ebitda margin crore crore", "ebitda margin crore", ["ebitda", "crore"])
    assert n == 3 and rate == pytest.approx(1 / 3)


G = [Figure("ebitda_margin", "FY27", 16, 17, "percent"), Figure("capex", "FY27", 1200, 1200, "inr_crore")]


def test_score_numbers_tolerance_and_wrong_inr():
    ext = [Figure("ebitda_margin", "FY27", 16.05, 17.05, "percent"),  # within 0.5%
           Figure("capex", "FY27", 120, 120, "inr_crore"),            # wrong ₹ figure
           Figure("pat", "FY27", 10, 10, "inr_crore")]
    s = score_numbers(ext, G)
    assert s.recall == 0.5 and s.precision == pytest.approx(1 / 3)
    assert [(e.metric, g.low) for e, g in s.wrong_inr] == [("capex", 1200)]


def test_bps_needs_exact_match():
    gold = [Figure("ebitda_margin", "FY27", 150, 150, "bps")]
    assert score_numbers([Figure("ebitda_margin", "FY27", 150.5, 150.5, "bps")], gold).recall == 0


def test_percentile_matches_numpy_linear():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 95) == 10
    assert percentile([], 50) is None


def test_gate_flags_recall_drop_and_wrong_inr():
    rows = [
        {"llm_model": "q", "llm_precision": "fp16", "numeric_recall": "0.90", "wrong_inr_figures": "0"},
        {"llm_model": "q", "llm_precision": "fp8", "numeric_recall": "0.89", "wrong_inr_figures": "0"},
        {"llm_model": "q", "llm_precision": "fp8", "numeric_recall": "0.85", "wrong_inr_figures": "0"},
        {"llm_model": "q", "llm_precision": "fp8", "numeric_recall": "0.90", "wrong_inr_figures": "1"},
        {"llm_model": "m", "llm_precision": "fp8", "numeric_recall": "0.90", "wrong_inr_figures": "0"},
    ]
    apply_gate(rows)
    assert [r["gate"] for r in rows] == ["PASS", "PASS", "FAIL", "FAIL", "no_fp16_baseline"]


def test_gpu_util_parsing_both_tools():
    amd = [{"gpu": 0, "usage": {"gfx_activity": {"value": 42, "unit": "%"}}},
           {"gpu": 1, "usage": {"gfx_activity": {"value": 8, "unit": "%"}}}]
    rocm = {"card0": {"GPU use (%)": "35"}}
    assert extract_utilisation(amd) == [42.0, 8.0]
    assert extract_utilisation(rocm) == [35.0]


def test_summarize_level_from_events():
    ev = [
        {"event": "asr_done", "call_id": "n5-0", "t": 1.0, "latency_s": 0.4},
        {"event": "llm_done", "call_id": "n5-0", "t": 3.0, "source_t": 1.0,
         "calls": [{"ttft_s": 0.1, "total_s": 1.5, "output_tok_per_s": 80, "completion_tokens": 100}]},
        {"event": "alert", "call_id": "n5-0", "t": 3.0, "latency_s": 2.0},
        {"event": "call_done", "call_id": "n5-0", "t": 4.0, "skipped_windows": 0},
        {"event": "alert", "call_id": "n1-0", "t": 9.0, "latency_s": 99.0},  # other level, ignored
    ]
    rows = {r["measure"]: r for r in summarize_level(ev, 5)}
    assert rows["alert_latency_s"]["p95"] == 2.0
    assert rows["insight_latency_s"]["p50"] == 2.0
    assert rows["throughput_output_tok_per_s"]["mean"] == pytest.approx(100 / 3)
