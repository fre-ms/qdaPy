"""The new measures, held to the plugin's numbers.

Same principle as ``test_congruence.py``: fixtures generated at random,
computed by the plugin's JavaScript, frozen here. The reference file is
byte-identical to the one in qdaR, so all three implementations answer to a
single set of numbers rather than to each other.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import qdapy

HERE = Path(__file__).parent
REF = json.loads((HERE / "qdaz-e35-reference.json").read_text(encoding="utf-8"))


def num(x):
    return math.nan if x is None else float(x)


def same(a, b):
    a, b = num(a), num(b)
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-12


def test_the_reference_covers_the_awkward_cases():
    assert len(REF["unitizing"]) >= 25
    widths = {len(f["fixture"]["coders"]) for f in REF["unitizing"]}
    assert {2, 3} <= widths, "two coders and more"
    # saturation that is reached and saturation that never is
    notations = {f["expected"]["notation"] for f in REF["saturation"]}
    assert None in notations and len(notations) > 2
    # proportions at both edges
    ks = {f["fixture"]["k"] for f in REF["wilson"]}
    assert 0 in ks


@pytest.mark.parametrize("i", range(len(REF["unitizing"])))
def test_unitizing_matches_the_plugin(i):
    f, e = REF["unitizing"][i]["fixture"], REF["unitizing"][i]["expected"]
    a = qdapy.unitizing_alpha(f["coders"])
    ident = qdapy.unitizing_alpha(f["coders"], lambda x, y: 0)
    assert same(a["alpha"] if a else None, e["alpha"])
    assert same(a["Do"] if a else None, e["Do"])
    assert same(a["De"] if a else None, e["De"])
    assert (a["intersections"] if a else None) == e["inter"]
    assert same(ident["alpha"] if ident else None, e["identAlpha"])
    assert same(qdapy.window_diff(f["coders"][0], f["coders"][1], f["length"]), e["wd"])
    assert same(qdapy.pk(f["coders"][0], f["coders"][1], f["length"]), e["pk"])


@pytest.mark.parametrize("i", range(len(REF["saturation"])))
def test_saturation_matches_the_plugin(i):
    f, e = REF["saturation"][i]["fixture"], REF["saturation"][i]["expected"]
    r = qdapy.saturation_ratio(f["counts"], base_size=f["base"],
                               run_length=f["run"], threshold=f["threshold"])
    assert r["notation"] == e["notation"]
    assert r["saturated_at"] == e["at"]
    assert same(r["base_codes"], e["base"])
    for j, ratio in enumerate(e["ratios"]):
        assert same(r["runs"].iloc[j]["ratio"], ratio), j


@pytest.mark.parametrize("i", range(len(REF["wilson"])))
def test_wilson_matches_the_plugin(i):
    f, e = REF["wilson"][i]["fixture"], REF["wilson"][i]["expected"]
    w = qdapy.wilson(f["k"], f["n"])
    assert same(w["estimate"], e["estimate"])
    assert same(w["lo"], e["lo"])
    assert same(w["hi"], e["hi"])


def test_the_reference_is_the_file_the_r_package_uses():
    ours = (HERE / "qdaz-e35-reference.json").read_bytes()
    theirs = HERE.parents[1] / "qdaR/tests/testthat/qdaz-e35-reference.json"
    if not theirs.exists():
        pytest.skip("qdaR sources not present")
    assert ours == theirs.read_bytes()
