"""E35: intervals, per-code agreement, and the diagnostics for a bad kappa."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import qdapy
from qdapy.reliability import mulberry32


def matrix(**cols):
    m = pd.DataFrame(cols, dtype=object)
    m.attrs["multi"] = 0
    return m


def test_the_paradox_indices_explain_a_collapsed_kappa():
    # Feinstein & Cicchetti's case: a=3, b=0, c=1, d=0; po = 3/4
    u = matrix(ann=["A", "A", "A", "B"], bob=["A", "A", "A", "A"])
    assert qdapy.kappa(u) == pytest.approx(0)
    d = qdapy.paradox(u)
    assert d["prevalence_index"] == pytest.approx(0.75)
    assert d["bias_index"] == pytest.approx(0.25)
    assert d["pabak"] == pytest.approx(0.5)
    assert d["table"] == {"a": 3, "b": 0, "c": 1, "d": 0}


def test_the_paradox_indices_refuse_cases_they_do_not_describe():
    assert qdapy.paradox(matrix(a=["A"], b=["A"], c=["A"])) is None
    assert qdapy.paradox(matrix(a=["A", "B"], b=["B", "C"])) is None


def test_balanced_marginals_leave_kappa_intact():
    bal = matrix(ann=["A", "A", "B", "B", "A"], bob=["A", "A", "B", "B", "B"])
    assert qdapy.paradox(bal)["prevalence_index"] < 0.25
    assert qdapy.kappa(bal) > 0.5


@pytest.mark.parametrize(("k", "n"), [(0, 10), (10, 10), (1, 3), (2, 40)])
def test_the_wilson_interval_stays_inside_the_unit_interval(k, n):
    v = qdapy.wilson(k, n)
    assert 0 <= v["lo"] <= v["estimate"] <= v["hi"] <= 1


def test_the_wilson_interval_matches_the_hand_computation():
    w = qdapy.wilson(2, 40)
    assert w["estimate"] == pytest.approx(0.05)
    assert w["lo"] == pytest.approx(0.01382067, abs=1e-7)
    assert w["hi"] == pytest.approx(0.16503877, abs=1e-7)


def test_a_degenerate_proportion_says_nan_rather_than_a_number():
    assert math.isnan(qdapy.wilson(1, 0)["estimate"])
    assert math.isnan(qdapy.wilson(5, 3)["estimate"])


def test_the_generator_reproduces_the_plugins_bit_for_bit():
    # this is the whole reason for not using random.Random: an interval
    # reported here must be the interval the plugin and qdaR report
    r = mulberry32(42)
    got = [round(r(), 12) for _ in range(6)]
    assert got == [0.601103751920, 0.448290558998, 0.852465793490,
                   0.669734041439, 0.174813898746, 0.526592542185]
    assert mulberry32(7)() == mulberry32(7)()
    assert mulberry32(1)() != mulberry32(2)()


def test_the_bootstrap_interval_brackets_the_estimate_and_repeats():
    u = matrix(ann=["A", "B"] * 30, bob=["A", "B", "B", "A"] * 15)
    ci = qdapy.bootstrap_ci(u, qdapy.kappa, resamples=300)
    assert ci["lo"] <= ci["estimate"] <= ci["hi"]
    assert ci == qdapy.bootstrap_ci(u, qdapy.kappa, resamples=300)
    other = qdapy.bootstrap_ci(u, qdapy.kappa, resamples=300, seed=7)
    assert (other["lo"], other["hi"]) != (ci["lo"], ci["hi"])


def test_too_little_data_gives_no_interval_rather_than_a_fake_one():
    assert qdapy.bootstrap_ci(matrix(a=["A"], b=["A"]), qdapy.kappa,
                              resamples=50) is None


def test_agreement_is_reported_per_code_with_its_prevalence():
    frag = pd.DataFrame({
        "annotationKey": [f"s{i}" for i in range(1, 7) for _ in range(2)],
        "codedBy": ["ann", "bob"] * 6,
        "code": ["A", "A", "A", "B", "B", "B", "A", "A", "B", "B", "A", "A"],
    })
    out = qdapy.agreement_by_code(frag, min_n=1)
    assert set(out["code"]) == {"A", "B"}
    assert ((out["lo"] <= out["prevalence"]) & (out["prevalence"] <= out["hi"])).all()
    assert list(out["n"]) == sorted(out["n"], reverse=True)
    assert len(qdapy.agreement_by_code(frag, min_n=99)) == 0
