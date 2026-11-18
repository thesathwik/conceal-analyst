import pytest

from compare import GuidanceTracker, Thresholds, classify, compare
from schema import AlertKind, Metric

TH = Thresholds(relative_pct=5.0, margin_bps=50.0)


def m(metric="capex", lo=1000.0, hi=None, unit="inr_crore", period="FY27", type_="guidance") -> Metric:
    return Metric(metric=metric, value_low=lo, value_high=lo if hi is None else hi, unit=unit,
                  period=period, type=type_, quote="q")


@pytest.mark.parametrize(
    "prior, new, kind",
    [
        (1000, 940, AlertKind.CUT),         # -6% relative
        (1000, 960, AlertKind.REITERATED),  # -4%
        (1000, 1060, AlertKind.RAISE),      # +6%
        (1000, 1050, AlertKind.REITERATED),  # exactly 5% is within threshold
    ],
)
def test_relative_threshold(prior, new, kind):
    assert classify(m(lo=new), m(lo=prior), TH).kind is kind


def test_margin_uses_bps():
    prior = m("ebitda_margin", 16, 17, unit="percent")
    # midpoint 16.5 -> 15.9 = -60 bps (only -3.6% relative, so the bps rule must decide)
    a = classify(m("ebitda_margin", 15.4, 16.4, unit="percent"), prior, TH)
    assert a.kind is AlertKind.CUT and a.change_bps == pytest.approx(-60)
    a = classify(m("ebitda_margin", 16.2, 17.2, unit="percent"), prior, TH)
    assert a.kind is AlertKind.REITERATED and a.change_bps == pytest.approx(20)


def test_new_when_no_prior_or_unit_mismatch():
    assert classify(m(), None, TH).kind is AlertKind.NEW
    assert classify(m(unit="percent", lo=10), m(), TH).kind is AlertKind.NEW


def test_thresholds_are_config():
    loose = Thresholds(relative_pct=10.0, margin_bps=100.0)
    assert classify(m(lo=940), m(lo=1000), loose).kind is AlertKind.REITERATED


def test_compare_matches_on_normalized_period_and_ignores_actuals():
    prior = [m(period="FY27", lo=1000)]
    alerts = compare([m(period="FY 2026-27", lo=900), m(lo=5, type_="actual")], prior, TH)
    assert [a.kind for a in alerts] == [AlertKind.CUT]
    assert alerts[0].period == "FY27"


def test_zero_prior_midpoint():
    assert classify(m("net_debt", 100), m("net_debt", 0), TH).kind is AlertKind.RAISE


def test_tracker_emits_only_on_change_and_reports_withdrawn():
    prior = [m(lo=1000), m("order_book", 25000)]
    tr = GuidanceTracker(prior, TH)
    first = tr.update([m(lo=900)])
    assert [a.kind for a in first] == [AlertKind.CUT]
    assert tr.update([m(lo=900)]) == []  # same sentence seen again in the next window
    assert [a.kind for a in tr.update([m(lo=1000)])] == [AlertKind.REITERATED]
    withdrawn = tr.withdrawn()
    assert [(a.kind, a.metric) for a in withdrawn] == [(AlertKind.WITHDRAWN, "order_book")]
    assert tr.withdrawn() == []


def test_reiterated_does_not_notify():
    assert not AlertKind.REITERATED.notifies
    assert AlertKind.CUT.notifies
