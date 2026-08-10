"""E37: reliability of the segmentation itself."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import qdapy

# Krippendorff, Content Analysis 3rd ed., replacement of section 12.4,
# Figure 12.11. The figure prints the five intersections behind the observed
# disagreement:
#   (15-15(1-0)) + (23-5(1-1)) + 2*3 + (10-5(1-1)) + (5-5(1-0)) = 39, N_o = 5
# so uD_o = 7.8. D_e is not checked: it depends on the individual unit
# lengths, which the figure does not print.
OBS_I = [{"start": 0, "end": 15, "value": 1}, {"start": 20, "end": 38, "value": 1},
         {"start": 60, "end": 65, "value": 2}, {"start": 70, "end": 75, "value": 1}]
OBS_J = [{"start": 0, "end": 15, "value": 1}, {"start": 33, "end": 43, "value": 3},
         {"start": 50, "end": 53, "value": 4}, {"start": 60, "end": 65, "value": 2},
         {"start": 70, "end": 80, "value": 5}]

REF = [{"start": 0, "end": 20}, {"start": 20, "end": 40}, {"start": 40, "end": 60}]
NEAR = [{"start": 0, "end": 22}, {"start": 22, "end": 40}, {"start": 40, "end": 60}]
FAR = [{"start": 0, "end": 40}, {"start": 40, "end": 60}]


def test_the_unitizing_alpha_reproduces_the_published_example():
    r = qdapy.unitizing_alpha([OBS_I, OBS_J])
    assert r["intersections"] == 5
    assert r["Do"] == pytest.approx(7.8, abs=1e-12)
    assert r["units"] == 9


def test_ignoring_the_categories_can_only_improve_agreement():
    nominal = qdapy.unitizing_alpha([OBS_I, OBS_J])
    ident = qdapy.unitizing_alpha([OBS_I, OBS_J], lambda a, b: 0)
    assert ident["Do"] < nominal["Do"]
    assert ident["alpha"] > nominal["alpha"]


def test_the_three_anchors_any_alpha_has():
    same = [{"start": 0, "end": 10, "value": "A"}]
    other = [{"start": 0, "end": 10, "value": "B"}]
    away = [{"start": 50, "end": 60, "value": "A"}]
    assert qdapy.unitizing_alpha([same, same])["alpha"] == pytest.approx(1)
    assert qdapy.unitizing_alpha([same, other])["alpha"] < 1
    # identification is perfect even when the coding is not
    assert qdapy.unitizing_alpha([same, other], lambda a, b: 0)["alpha"] == pytest.approx(1)
    assert qdapy.unitizing_alpha([same, away])["alpha"] < 0


def test_one_coder_is_not_a_reliability_test():
    assert qdapy.unitizing_alpha([[{"start": 0, "end": 5, "value": "A"}]]) is None


def test_identical_segmentations_score_zero_error():
    assert qdapy.window_diff(REF, REF, 60) == 0
    assert qdapy.pk(REF, REF, 60) == 0


def test_a_missing_boundary_costs_more_than_a_shifted_one():
    assert qdapy.window_diff(REF, NEAR, 60) > 0
    assert qdapy.window_diff(REF, FAR, 60) > qdapy.window_diff(REF, NEAR, 60)
    assert qdapy.pk(REF, FAR, 60) > qdapy.pk(REF, NEAR, 60)


def test_a_continuum_too_short_for_a_window_has_no_answer():
    assert math.isnan(qdapy.window_diff(REF, REF, 1))
    assert math.isnan(qdapy.pk(REF, REF, 1))


def test_segments_come_out_of_the_position_columns_per_coder():
    frag = pd.DataFrame({
        "codedBy": ["ann", "ann", "bob"], "code": ["A", "B", "A"],
        "positionKind": ["text"] * 3,
        "positionStart": [0, 30, 5], "positionEnd": [20, 50, 22],
    })
    segs = qdapy.segments(frag)
    assert list(segs) == ["ann", "bob"]
    assert [s["start"] for s in segs["ann"]] == [0, 30]


def test_pdf_segments_are_dropped_with_a_warning_not_approximated():
    frag = pd.DataFrame({
        "codedBy": ["ann", "bob"], "code": ["A", "A"],
        "positionKind": ["text", "pdf"],
        "positionStart": [0, None], "positionEnd": [20, None],
    })
    with pytest.warns(UserWarning, match="PDF"):
        segs = qdapy.segments(frag)
    assert list(segs) == ["ann"]


def test_an_export_predating_the_position_columns_says_so():
    with pytest.raises(KeyError, match="predates the position columns"):
        qdapy.segments(pd.DataFrame({"codedBy": ["ann"], "code": ["A"]}))
