import json
from pathlib import Path

import pytest

from extract import ground_and_normalize, parse_json_object
from schema import Metric, PriorGuidance, parse_extraction

ROOT = Path(__file__).resolve().parent.parent

GOOD = {
    "metric": "ebitda_margin", "value_low": 16.0, "value_high": 17.0, "unit": "percent",
    "period": "FY27", "type": "guidance", "quote": "we expect EBITDA margins of 16 to 17 percent for FY27",
}


def test_example_from_claude_md_parses():
    parsed = parse_extraction({"metrics": [GOOD], "tone_flags": [{"flag": "hedging", "quote": "x"}]})
    assert parsed.extraction.metrics[0].midpoint == 16.5
    assert parsed.rejected == []


def test_bad_items_are_rejected_individually():
    bad = {**GOOD, "metric": "ebitda"}  # not in the enum
    parsed = parse_extraction({"metrics": [GOOD, bad, {"metric": "pat"}], "tone_flags": []})
    assert len(parsed.extraction.metrics) == 1
    assert len(parsed.rejected) == 2


@pytest.mark.parametrize("payload", [[], "x", {"metrics": "nope"}])
def test_wrong_top_level_shape_raises(payload):
    with pytest.raises(ValueError):
        parse_extraction(payload)


def test_reversed_range_is_reordered():
    assert Metric(**{**GOOD, "value_low": 17, "value_high": 16}).value_low == 16


def test_parse_json_object_tolerates_fences_and_chatter():
    assert parse_json_object('```json\n{"metrics": []}\n```') == {"metrics": []}
    assert parse_json_object('Sure! {"metrics": [], "tone_flags": []} hope this helps') == {
        "metrics": [], "tone_flags": []}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_ground_and_normalize_drops_hallucinated_quotes():
    window = "For FY 2026-27, we expect EBITDA margins of 16 to 17 percent for FY27, broadly."
    good = Metric(**{**GOOD, "period": "FY 2026-27"})
    hallucinated = Metric(**{**GOOD, "quote": "we expect EBITDA margins of 18 percent"})
    parsed = parse_extraction({"metrics": [good.model_dump(), hallucinated.model_dump()]})
    out, rejected = ground_and_normalize(parsed.extraction, window)
    assert [x.period for x in out.metrics] == ["FY27"]
    assert rejected[0]["error"] == "quote_not_in_window"


def test_last_quarter_file_is_valid():
    data = json.loads((ROOT / "data" / "last_quarter.json").read_text())
    assert PriorGuidance.model_validate(data).metrics


def test_prior_guidance_rejects_actuals():
    with pytest.raises(ValueError):
        PriorGuidance.model_validate({"metrics": [{**GOOD, "type": "actual"}]})


def test_gold_numbers_file_shape():
    data = json.loads((ROOT / "eval" / "gold_numbers.json").read_text())
    for f in data["figures"]:
        Metric.model_validate({**f, "quote": "gold"})
