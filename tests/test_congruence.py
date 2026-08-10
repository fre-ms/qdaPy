"""qdaPy, qdaR and the qdaZ plugin must compute the same coefficients.

Two implementations that agree on a hand-picked example prove very little.
These fixtures were generated at random -- varying coders, units, categories,
deliberately skewed marginals and missing ratings -- run through the plugin's
JavaScript, and the results frozen.  ``qdaz-reference.json`` is byte-identical
to the file in the R package, so all three implementations are held to one
reference rather than to each other.

If a coefficient ever moves in one implementation and not the others, this
fails.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qdapy

REFERENCE = json.loads(
    (Path(__file__).parent / "qdaz-reference.json").read_text(encoding="utf-8")
)

MEASURES = ("percent", "cohen", "brennan", "fleiss", "alpha", "ac1")


def as_matrix(units) -> pd.DataFrame:
    rows = [[None if v is None else str(v) for v in row] for row in units]
    m = pd.DataFrame(
        np.array(rows, dtype=object),
        columns=[f"coder{i + 1}" for i in range(len(rows[0]))],
    )
    m.attrs["multi"] = 0
    return m


def number(value) -> float:
    return math.nan if value is None else float(value)


def computed(m: pd.DataFrame) -> dict[str, float]:
    two = m.shape[1] == 2
    return {
        "percent": qdapy.percent_agreement(m),
        "cohen": qdapy.kappa(m) if two else math.nan,
        "brennan": qdapy.brennan(m) if two else math.nan,
        "fleiss": qdapy.fleiss(m),
        "alpha": qdapy.alpha(m),
        "ac1": qdapy.ac1(m),
    }


def test_the_frozen_reference_covers_the_awkward_cases():
    assert len(REFERENCE) > 50
    widths = {len(f["units"][0]) for f in REFERENCE}
    assert {2, 3, 4} <= widths                       # two coders and more
    with_gaps = sum(
        1 for f in REFERENCE
        if any(v is None for row in f["units"] for v in row)
    )
    assert with_gaps > 20                            # skipped ratings
    # and the values span the range where the coefficients start to disagree
    fleiss = [number(f["expected"]["fleiss"]) for f in REFERENCE]
    finite = [v for v in fleiss if not math.isnan(v)]
    assert min(finite) < 0.05
    assert max(finite) > 0.3


@pytest.mark.parametrize("index", range(len(REFERENCE)))
def test_every_coefficient_matches_the_plugin(index):
    fixture = REFERENCE[index]
    m = as_matrix(fixture["units"])
    got = computed(m)
    for measure in MEASURES:
        want = number(fixture["expected"].get(measure))
        have = got[measure]
        if math.isnan(want):
            assert math.isnan(have), f"{measure}: plugin nan, qdaPy {have}"
        else:
            assert have == pytest.approx(want, abs=1e-12), measure


def test_the_reference_is_the_same_file_the_r_package_uses():
    ours = (Path(__file__).parent / "qdaz-reference.json").read_bytes()
    theirs = Path(__file__).parents[2] / "qdaR/tests/testthat/qdaz-reference.json"
    if not theirs.exists():          # the R twin is not checked out beside us
        pytest.skip("qdaR sources not present")
    assert ours == theirs.read_bytes()


def test_undefined_cases_are_undefined_in_both_implementations():
    undefined = sum(
        1
        for f in REFERENCE
        for measure in MEASURES
        if math.isnan(number(f["expected"].get(measure)))
    )
    # the three- and four-coder fixtures leave Cohen and Brennan undefined
    assert undefined > 0
