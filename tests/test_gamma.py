"""E37.4: gamma, and the alignment it is measured against."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import qdapy
from qdapy.gamma import best_alignment, dissimilarity, unitary_disorder

HERE = Path(__file__).parent
REF = json.loads((HERE / "qdaz-gamma-reference.json").read_text(encoding="utf-8"))


def U(s, e, v):
    return {"start": s, "end": e, "value": v}


BEST = [[U(0, 10, "A"), U(20, 30, "B"), U(40, 50, "C")]] * 3
MIDDLE = [
    [U(0, 10, "A"), U(20, 30, "A"), U(40, 50, "C")],
    [U(0, 10, "A"), U(20, 30, "B"), U(40, 50, "C")],
    [U(22, 32, "B"), U(40, 50, "C")],
]
WORST = [[U(0, 10, "A")], [U(20, 30, "B")], [U(40, 50, "C")]]


def test_the_positional_dissimilarity_matches_equation_three():
    # ((|0-2| + |10-12|) / ((10-0) + (12-2)))^2 = (4/20)^2 = 0.04
    assert dissimilarity(U(0, 10, "A"), U(2, 12, "A")) == pytest.approx(0.04)
    assert dissimilarity(U(0, 10, "A"), U(0, 10, "A")) == 0


def test_position_and_category_are_added_not_traded_off():
    # right place, wrong code costs exactly the category term
    assert dissimilarity(U(0, 10, "A"), U(0, 10, "B")) == pytest.approx(1)
    # and a partial category distance is honoured
    assert dissimilarity(U(0, 10, "A"), U(0, 10, "B"),
                         dist_cat=lambda a, b: 0.5) == pytest.approx(0.5)


def test_an_unaligned_unit_costs_delta_empty():
    assert dissimilarity(U(0, 10, "A"), None) == 1
    assert dissimilarity(None, None) == 1
    # so a unitary alignment holding one real unit costs exactly that
    assert unitary_disorder([U(0, 10, "A"), None, None]) == pytest.approx(1)
    assert unitary_disorder([U(0, 10, "A")] * 3) == 0


def test_the_three_configurations_of_figure_eleven():
    g = lambda d: qdapy.gamma(d, samples=20, seed=42)  # noqa: E731
    best, middle, worst = g(BEST), g(MIDDLE), g(WORST)
    assert best["gamma"] == pytest.approx(1)
    assert best["observed"] == pytest.approx(0)
    assert worst["gamma"] < 0          # worse than annotating at random
    assert worst["gamma"] < middle["gamma"] < best["gamma"]


def test_the_worst_case_disorder_by_hand():
    # three lone units: each unitary alignment costs Delta_empty, and the
    # mean number of units per annotator is 1, so the disorder is 3
    assert qdapy.gamma(WORST, samples=5)["observed"] == pytest.approx(3)


def test_gamma_reports_the_alignment_it_found():
    r = qdapy.gamma(BEST, samples=10)
    assert len(r["alignment"]) == 3
    assert all(len(ua) == 3 for ua in r["alignment"])


def test_the_result_is_reproducible_and_the_seed_matters():
    a = qdapy.gamma(MIDDLE, samples=20, seed=42)
    assert a["gamma"] == qdapy.gamma(MIDDLE, samples=20, seed=42)["gamma"]
    assert a["gamma"] != qdapy.gamma(MIDDLE, samples=20, seed=7)["gamma"]


def test_one_annotator_is_not_an_agreement_question():
    assert qdapy.gamma([[U(0, 10, "A")]]) is None


def test_refusing_beats_approximating():
    # a gamma produced by a heuristic is not gamma, so an exhausted search
    # returns nan with a reason rather than the best thing it happened to see
    r = qdapy.gamma(MIDDLE, samples=5, max_nodes=1)
    assert r["exhausted"]
    assert math.isnan(r["gamma"])
    assert "exact" in r["reason"]


def test_the_pruning_theorem_discards_candidates():
    dense = [
        [U(0, 10, "A"), U(12, 22, "A"), U(24, 34, "A")],
        [U(0, 10, "A"), U(12, 22, "A"), U(60, 70, "B")],
    ]
    r = qdapy.gamma(dense, samples=5)
    assert r["candidates"] < (3 + 1) * (3 + 1) - 1


def test_the_sampling_rule_says_how_many_samples_would_be_enough():
    r = qdapy.gamma(MIDDLE, samples=20, seed=42)
    assert r["recommended_samples"] >= 0
    assert r["samples"] == 20


def test_a_best_alignment_can_pair_units_that_do_not_overlap():
    # what alpha_U cannot express: the configuration says these belong
    # together even though they miss each other
    data = [[U(0, 10, "A")], [U(11, 21, "A")]]
    r = best_alignment(data)
    assert len(r["alignment"]) == 1
    assert len(r["alignment"][0]) == 2


@pytest.mark.parametrize("i", range(len(REF["cases"])))
def test_gamma_matches_the_plugin_on_every_fixture(i):
    case = REF["cases"][i]
    r = qdapy.gamma(case["fixture"]["coders"], samples=12, seed=42)
    e = case["expected"]
    for key in ("gamma", "observed", "expected"):
        assert r[key] == pytest.approx(e[key], abs=1e-12), key
    assert r["candidates"] == e["candidates"]


def test_the_reference_is_the_file_the_r_package_uses():
    ours = (HERE / "qdaz-gamma-reference.json").read_bytes()
    theirs = HERE.parents[1] / "qdaR/tests/testthat/qdaz-gamma-reference.json"
    if not theirs.exists():
        pytest.skip("qdaR sources not present")
    assert ours == theirs.read_bytes()
