"""E36.2: the accumulation models behind the saturation index."""

from __future__ import annotations

import math

import pytest

import qdapy

FLATTENING = [8, 13, 16, 18, 19, 20, 20, 21]
LINEAR = [5, 10, 15, 20, 25, 30, 35, 40]


def test_the_direct_iw_estimate_is_exact_algebra_from_the_end_points():
    # Lowe et al. (a16): A and b follow from T_1 and T_N alone
    N, T1, TN = 8, 8, 21
    A = (N - 1) * T1 * TN / (N * T1 - TN)
    b = (TN - N * T1) / ((1 - N) * TN)
    assert A * b * N / (1 + b * (N - 1)) == pytest.approx(21)
    assert A * b == pytest.approx(8)


def test_a_flattening_curve_scores_high_and_a_linear_one_low():
    flat = qdapy.saturation_index(FLATTENING, "IW")
    lin = qdapy.saturation_index(LINEAR, "IW")
    assert flat["index"] > 70
    assert lin["index"] < 20
    assert lin["A"] > 100        # a straight line implies much unseen


def test_all_three_models_fit_and_need_not_agree():
    fits = [qdapy.saturation_index(FLATTENING, m) for m in qdapy.discovery.MODELS]
    for f in fits:
        assert math.isfinite(f["A"])
        assert 0 < f["b"] < 1
        assert f["rmse"] < 1
        assert len(f["fitted"]) == len(FLATTENING)
    # that they differ is information, not a defect
    values = [f["A"] for f in fits]
    assert max(values) - min(values) > 1


def test_the_index_is_the_share_of_the_floored_estimate():
    f = qdapy.saturation_index(FLATTENING, "IW")
    assert f["index"] == 100 * 21 / math.floor(f["A"])


def test_the_sw_model_survives_numbers_that_would_overflow_a_gamma():
    y = list(range(20, 271))     # long enough for gamma(1 - b + n) to blow up
    f = qdapy.saturation_index(y, "SW")
    assert math.isfinite(f["A"])
    assert all(math.isfinite(v) for v in f["fitted"])


def test_the_model_predicts_how_much_more_material_a_target_needs():
    f = qdapy.saturation_index(FLATTENING, "IW")
    assert qdapy.documents_for(f, 90) <= qdapy.documents_for(f, 95)
    assert qdapy.documents_for(f, 95) >= 8


def test_a_target_the_model_never_reaches_says_none():
    f = qdapy.saturation_index(LINEAR, "IW")
    assert qdapy.documents_for(f, 99, max_n=20) is None


def test_too_little_material_is_refused_with_a_reason():
    r = qdapy.saturation_index([1, 2])
    assert math.isnan(r["index"])
    assert "fewer than three" in r["reason"]
    assert "no themes" in qdapy.saturation_index([0, 0, 0, 0])["reason"]


def test_r_and_python_agree_on_the_fitted_curve():
    # the R twin reports A = 26.80, b = 0.320, index 80.8 %, n = 25 for 95 %
    f = qdapy.saturation_index(FLATTENING, "IW")
    assert round(f["A"], 2) == 26.80
    assert round(f["b"], 3) == 0.320
    assert round(f["index"], 1) == 80.8
    assert qdapy.documents_for(f, 95) == 25
