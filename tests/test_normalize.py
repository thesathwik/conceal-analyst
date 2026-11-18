import pytest

from normalize import (
    Span,
    normalize_for_match,
    normalize_period,
    parse_bps,
    parse_inr_crore,
    parse_percent,
    quote_in_window,
    reconcile,
    to_float,
)
from schema import Metric


@pytest.mark.parametrize(
    "text, expected",
    [
        ("₹1,250 crore", [Span(1250, 1250)]),
        ("12.5 billion rupees", [Span(1250, 1250)]),
        ("about 450 crores", [Span(450, 450)]),
        ("Rs. 300 million", [Span(30, 30)]),
        ("INR 75 lakh", [Span(0.75, 0.75)]),
        ("an order book of 1.2 lakh crore", [Span(120000, 120000)]),
        ("capex of 5 thousand crore", [Span(5000, 5000)]),
        ("between 1,200 and 1,300 crore", [Span(1200, 1300)]),
        ("₹1,200 to ₹1,300 crores", [Span(1200, 1300)]),
        ("1,25,000 crore", [Span(125000, 125000)]),
        ("revenue of 980 Cr", [Span(980, 980)]),
        ("we did 450 crore against 400 crore", [Span(450, 450), Span(400, 400)]),
    ],
)
def test_parse_inr_crore(text, expected):
    assert parse_inr_crore(text) == expected


@pytest.mark.parametrize("text", ["$1.2 billion", "USD 300 million", "2 billion dollars"])
def test_foreign_currency_is_not_rupees(text):
    assert parse_inr_crore(text) == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("EBITDA margins of 16 to 17 percent", [Span(16, 17)]),
        ("16-17%", [Span(16, 17)]),
        ("16%-17%", [Span(16, 17)]),
        ("growth of 12.5%", [Span(12.5, 12.5)]),
        ("around 18 per cent", [Span(18, 18)]),
        ("between 14 and 15 percent", [Span(14, 15)]),
    ],
)
def test_parse_percent(text, expected):
    assert parse_percent(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("150 basis points", [Span(150, 150)]),
        ("an improvement of 50 bps", [Span(50, 50)]),
        ("100 to 150 basis points", [Span(100, 150)]),
    ],
)
def test_parse_bps(text, expected):
    assert parse_bps(text) == expected


def test_indian_grouping():
    assert to_float("1,25,000") == 125000.0
    assert to_float("1,250.75") == 1250.75


def _m(**kw) -> Metric:
    base = dict(metric="capex", value_low=1250, value_high=1250, unit="inr_crore", period="FY27",
                type="guidance", quote="capex of 12.5 billion rupees")
    return Metric(**{**base, **kw})


def test_reconcile_keeps_correct_value():
    m, reason = reconcile(_m())
    assert reason == "ok" and m.value_low == 1250


def test_reconcile_overrides_llm_arithmetic():
    # LLM mis-converted 12.5 billion as 125 crore; the quote wins.
    m, reason = reconcile(_m(value_low=125, value_high=125))
    assert reason == "corrected_from_quote"
    assert (m.value_low, m.value_high) == (1250, 1250)


def test_reconcile_rejects_figure_not_in_quote():
    m, reason = reconcile(_m(quote="capex will be broadly similar to last year"))
    assert m is None and reason == "no_inr_crore_figure_in_quote"


def test_reconcile_rejects_ambiguous():
    m, _ = reconcile(_m(value_low=999, value_high=999, quote="450 crore this year versus 400 crore last year"))
    assert m is None


def test_reconcile_picks_matching_span_among_many():
    m, reason = reconcile(_m(value_low=450, value_high=450, quote="450 crore this year versus 400 crore last year"))
    assert reason == "ok" and m.value_low == 450


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("FY27", "FY27"), ("fy 27", "FY27"), ("FY'27", "FY27"), ("FY2027", "FY27"),
        ("FY 2026-27", "FY27"), ("Q3FY27", "Q3FY27"), ("Q3 FY27", "Q3FY27"), ("H2 FY27", "H2FY27"),
        ("2H FY 2027", "H2FY27"), ("next_year", "NEXT_YEAR"),
    ],
)
def test_normalize_period(raw, expected):
    assert normalize_period(raw) == expected


def test_quote_grounding_tolerates_punctuation_and_case():
    window = "So, for FY27 we expect EBITDA margins of 16 to 17 percent. Thank you."
    assert quote_in_window("we expect EBITDA margins of 16 to 17 percent", window)
    assert quote_in_window("We expect ebitda margins of 16 to 17 percent.", window)
    assert not quote_in_window("we expect EBITDA margins of 18 percent", window)
    assert not quote_in_window("", window)


def test_normalize_for_match_keeps_decimals_and_drops_grouping():
    assert normalize_for_match("₹1,250.5 crore.") == "₹1250.5 crore"
