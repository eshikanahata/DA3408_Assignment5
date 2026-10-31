"""test_gate.py -- tests for the pure decide() function.  Run: pytest -q test_gate.py"""
import pytest

from eta_mlflow import decide

PROD = dict(mae=10.0, mae_rush=10.0, mae_rain=10.0, p95_ms=2.0)


def cand(**kw):
    base = dict(mae=8.0, mae_rush=8.0, mae_rain=8.0, p95_ms=5.0)
    base.update(kw)
    return base


def test_clear_promote():
    r = decide(PROD, cand())
    assert r["promote"] is True and all(r["criteria"].values())
    assert set(r) == {"criteria", "promote"}


def test_gain_boundary_just_inside_and_outside():
    assert decide(PROD, cand(mae=9.7))["promote"] is True                   # exactly 3% better
    r = decide(PROD, cand(mae=9.71))                                        # 2.9% better
    assert r["promote"] is False and r["criteria"]["overall_mae_gain"] is False


def test_worse_overall_rejected():
    assert decide(PROD, cand(mae=11.0))["promote"] is False


def test_latency_boundary():
    assert decide(PROD, cand(p95_ms=10.0))["promote"] is True
    r = decide(PROD, cand(p95_ms=10.01))
    assert r["promote"] is False and r["criteria"]["latency_p95"] is False
    assert r["criteria"]["overall_mae_gain"] is True                        # only latency blocks it


def test_rush_slice_regression_boundary():
    assert decide(PROD, cand(mae_rush=10.5))["promote"] is True              # +5.0% allowed
    r = decide(PROD, cand(mae_rush=10.51))                                  # +5.1% not allowed
    assert r["promote"] is False and r["criteria"]["no_regression_rush"] is False
    assert r["criteria"]["no_regression_rain"] is True


def test_rain_slice_regression_boundary():
    assert decide(PROD, cand(mae_rain=10.5))["promote"] is True
    r = decide(PROD, cand(mae_rain=10.51))
    assert r["promote"] is False and r["criteria"]["no_regression_rain"] is False
    assert r["criteria"]["no_regression_rush"] is True


def test_better_overall_but_one_slice_regresses():
    # classic trap: average improves a lot, a slice gets much worse -> must NOT promote
    r = decide(PROD, cand(mae=5.0, mae_rush=14.0))
    assert r["criteria"]["overall_mae_gain"] is True and r["promote"] is False


def test_custom_thresholds_change_outcome():
    c = cand(mae=8.0)                                                       # 20% gain
    assert decide(PROD, c, min_rel_gain=0.19)["promote"] is True
    assert decide(PROD, c, min_rel_gain=0.21)["promote"] is False
    assert decide(PROD, c, min_rel_gain=0.60)["promote"] is False
    assert decide(PROD, cand(mae_rush=11.0), max_slice_regression=0.10)["promote"] is True
    assert decide(PROD, cand(p95_ms=20.0), max_p95_ms=25.0)["promote"] is True


def test_inputs_not_mutated():
    p, c = dict(PROD), cand()
    p0, c0 = dict(p), dict(c)
    decide(p, c)
    assert p == p0 and c == c0
