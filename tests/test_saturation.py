"""E36.1 and E35.5: saturation as a number, and drift over time."""

from __future__ import annotations

import pandas as pd
import pytest

import qdapy


def log(codes, users=None, docs=None):
    n = len(codes)
    return pd.DataFrame({
        "ts": [f"2026-01-{d:02d}T09:00:00Z" for d in range(1, n + 1)],
        "user": users or ["ann"] * n,
        "action": ["add"] * n,
        "code": codes,
        "citekey": docs or ["d1"] * n,
    })


def test_new_codes_are_counted_per_document_in_coding_order():
    h = log(["A", "B", "A", "C", "A", "B"],
            docs=["d1", "d1", "d2", "d2", "d3", "d3"])
    n = qdapy.new_codes(h)
    assert list(n["document"]) == ["d1", "d2", "d3"]
    assert list(n["new_codes"]) == [2, 1, 0]
    assert list(n["cumulative"]) == [2, 3, 3]


def test_the_saturation_ratio_follows_guest_namey_and_chen():
    r = qdapy.saturation_ratio([4, 3, 2, 1, 1, 0, 0, 0])
    assert r["base_codes"] == 10
    assert r["notation"] == "5+2"
    assert r["saturated_at"] == 5
    assert r["runs"].iloc[0]["ratio"] == pytest.approx(0.1)
    assert not r["runs"].iloc[0]["ok"]
    assert r["runs"].iloc[1]["ok"]


def test_a_stricter_threshold_never_declares_saturation_earlier():
    strict = qdapy.saturation_ratio([4, 3, 2, 1, 1, 1, 0, 0], threshold=0)
    lax = qdapy.saturation_ratio([4, 3, 2, 1, 1, 1, 0, 0], threshold=0.2)
    assert strict["saturated_at"] >= lax["saturated_at"]


def test_material_that_keeps_producing_codes_never_saturates():
    r = qdapy.saturation_ratio([4] * 8)
    assert r["notation"] is None
    assert r["saturated_at"] is None


def test_a_question_the_data_cannot_answer_says_why():
    assert qdapy.saturation_ratio([1, 2])["notation"] is None
    assert "too few" in qdapy.saturation_ratio([1, 2])["reason"]
    assert "base" in qdapy.saturation_ratio([0] * 8)["reason"]


def test_the_run_length_travels_with_the_number():
    # "5+2" and "5+3" are different claims
    assert qdapy.saturation_ratio([4, 3, 2, 1, 0, 0, 0],
                                  run_length=3)["notation"] == "4+3"


def test_a_coder_who_changed_everything_has_a_distance_of_one():
    d = qdapy.code_drift(log(["A"] * 8 + ["B"] * 8), windows=2)
    assert list(d["distance"]) == [0, 1]
    assert list(d["n"]) == [8, 8]


def test_a_steady_coder_shows_little_drift_and_coders_stay_apart():
    h = log(["A", "B"] * 6, users=["ann"] * 6 + ["bob"] * 6)
    d = qdapy.code_drift(h, windows=3)
    assert set(d["coder"]) == {"ann", "bob"}
    assert d["distance"].max() < 0.35


def test_an_empty_log_gives_an_empty_table_not_an_error():
    empty = pd.DataFrame({"ts": [], "user": [], "action": [], "code": [],
                          "citekey": []})
    assert len(qdapy.code_drift(empty)) == 0
    assert len(qdapy.new_codes(empty)) == 0
