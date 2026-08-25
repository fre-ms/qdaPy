"""The inferential layer, and its congruence with the R twin.

``stats-reference.json`` holds what qdaR computes for ``stats-fixture.csv``: a
few hundred codings over six codes, five documents and two coders, with
expected counts small enough to force the exact-test branch.  The two packages
implement the same statistics through entirely different libraries -- R's
``chisq.test``, ``cmdscale``, ``hclust`` and ``MASS::corresp`` against scipy
and hand-written linear algebra -- so agreeing to twelve decimal places is
evidence about the statistics rather than about shared code.

Two things are compared invariantly rather than literally: scaling coordinates
and correspondence scores are only defined up to sign and rotation, so the
pairwise distances between points are compared instead of the coordinates; and
the Monte Carlo p-value is random by construction in both packages, so it is
only required to be close.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

import qdapy

HERE = Path(__file__).parent
REF = json.loads((HERE / "stats-reference.json").read_text(encoding="utf-8"))
FIXTURE = pd.read_csv(HERE / "stats-fixture.csv", dtype=str)
CODES = REF["codes"]


def test_the_fixture_is_the_one_the_reference_was_made_from():
    assert len(FIXTURE) == REF["n_rows"]
    assert sorted(FIXTURE["code"].unique()) == sorted(CODES)


def test_chi_squared_matches_the_r_twin():
    res = qdapy.chisq(FIXTURE, "citekey")
    want = REF["chisq"]
    assert res.statistic == pytest.approx(want["statistic"], abs=1e-12)
    assert res.cramers_v == pytest.approx(want["cramers_v"], abs=1e-12)
    assert res.n == want["n"]
    assert res.expected_ok == want["expected_ok"]


def test_the_exact_branch_is_taken_and_says_so():
    res = qdapy.chisq(FIXTURE, "citekey")
    assert not res.expected_ok
    # a six-by-five table is beyond the exact test, so it must simulate
    assert "Monte Carlo" in res.test
    assert "seed" in res.test


def test_the_simulated_p_value_is_close_to_the_r_one():
    res = qdapy.chisq(FIXTURE, "citekey")
    # two independent Monte Carlo runs, so identity is not to be expected
    assert abs(res.p_value - REF["chisq"]["p_value"]) < 0.03


def test_the_simulated_p_value_is_reproducible():
    a = qdapy.chisq(FIXTURE, "citekey", resamples=300, seed=7)
    b = qdapy.chisq(FIXTURE, "citekey", resamples=300, seed=7)
    c = qdapy.chisq(FIXTURE, "citekey", resamples=300, seed=8)
    assert a.p_value == b.p_value
    assert a.p_value != c.p_value or a.p_value in (0.0, 1.0)


def test_a_two_by_two_table_gets_fishers_exact_test():
    frag = pd.DataFrame({
        "annotationKey": [f"s{i}" for i in range(12)],
        "citekey": ["doc1"] * 6 + ["doc2"] * 6,
        "code": list("AAAABABBBBAB"),
    })
    res = qdapy.chisq(frag, "citekey")
    assert not res.expected_ok
    assert "Fisher" in res.test


def test_a_table_with_healthy_counts_stays_asymptotic():
    frag = pd.DataFrame({
        "annotationKey": [f"s{i}" for i in range(80)],
        "citekey": (["doc1"] * 40) + (["doc2"] * 40),
        "code": (["A"] * 20 + ["B"] * 20) * 2,
    })
    res = qdapy.chisq(frag, "citekey")
    assert res.expected_ok
    assert res.test == "chi-squared test of independence"


def test_chi_squared_needs_two_codes_and_two_groups():
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s2"], "citekey": ["d1", "d1"],
        "code": ["A", "A"],
    })
    with pytest.raises(ValueError, match="at least two codes"):
        qdapy.chisq(frag, "citekey")


def test_a_group_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="one value per fragment"):
        qdapy.chisq(FIXTURE, pd.Series(["a", "b"]))


def test_jaccard_distances_match_the_r_twin():
    d = qdapy.code_distance(FIXTURE, min_n=3).loc[CODES, CODES]
    want = np.array(REF["distance"], dtype=float)
    assert np.abs(d.to_numpy() - want).max() < 1e-12


def test_distances_are_a_proper_distance_matrix():
    d = qdapy.code_distance(FIXTURE, min_n=3)
    m = d.to_numpy()
    assert np.allclose(np.diag(m), 0)
    assert np.allclose(m, m.T)
    assert m.min() >= 0
    assert m.max() <= 1


def test_min_n_actually_drops_rare_codes():
    d_all = qdapy.code_distance(FIXTURE, min_n=1)
    d_common = qdapy.code_distance(FIXTURE, min_n=30)
    assert len(d_common) < len(d_all)


def test_too_few_codes_for_a_distance_matrix_is_an_error():
    with pytest.raises(ValueError, match="fewer than two codes"):
        qdapy.code_distance(FIXTURE, min_n=10_000)


def test_the_code_map_matches_the_r_twin_up_to_sign():
    result = qdapy.mds(FIXTURE, min_n=3)
    points = result.points.set_index("code").loc[CODES, ["dim1", "dim2"]]
    got = squareform(pdist(points.to_numpy()))
    want = np.array(REF["mds_distance"], dtype=float)
    assert np.abs(got - want).max() < 1e-12


def test_the_clustering_matches_the_r_twin():
    result = qdapy.cluster(FIXTURE, min_n=3)
    assert result.cophenetic == pytest.approx(REF["cophenetic"], abs=1e-12)
    assert result.labels == CODES


def test_the_cophenetic_correlation_is_undefined_for_two_codes():
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s1", "s2", "s3"],
        "code": ["A", "B", "A", "B"],
    })
    result = qdapy.cluster(frag, min_n=1)
    # a single distance has no variance to correlate against
    assert math.isnan(result.cophenetic)


def test_the_tree_can_be_cut_into_groups():
    result = qdapy.cluster(FIXTURE, min_n=3)
    flat = result.flat(3)
    assert len(flat) == len(CODES)
    assert flat["cluster"].nunique() == 3


def test_correspondence_analysis_matches_the_r_twin():
    result = qdapy.ca(FIXTURE)
    assert result.total_inertia == pytest.approx(REF["ca"]["total_inertia"],
                                                 abs=1e-12)
    assert list(result.inertia) == pytest.approx(REF["ca"]["inertia"], abs=1e-12)
    assert list(result.inertia_share) == pytest.approx(
        REF["ca"]["inertia_share"], abs=1e-12)


def test_the_inertia_share_is_relative_to_the_whole_table():
    result = qdapy.ca(FIXTURE)
    # a six-by-five table has four dimensions; keeping two must not report 100%
    assert sum(result.inertia_share) < 1
    full = qdapy.ca(FIXTURE, n_dims=4)
    assert sum(full.inertia_share) == pytest.approx(1)


def test_total_inertia_is_the_chi_squared_statistic_over_n():
    result = qdapy.ca(FIXTURE)
    chi = qdapy.chisq(FIXTURE, "citekey")
    assert result.total_inertia == pytest.approx(chi.statistic / chi.n)


def test_correspondence_analysis_needs_a_two_by_two_table():
    frag = pd.DataFrame({
        "annotationKey": ["s1"], "citekey": ["d1"], "code": ["A"],
    })
    with pytest.raises(ValueError, match="2x2"):
        qdapy.ca(frag)


def test_the_agreement_measures_match_the_r_twin_on_the_same_fixture():
    row = qdapy.agreement.agreement(qdapy.units(FIXTURE)).iloc[0]
    for measure in ("percent", "cohen", "brennan", "fleiss", "alpha", "ac1"):
        want = REF["agreement"][measure]
        got = float(row[measure])
        if want is None:
            assert math.isnan(got), measure
        else:
            assert got == pytest.approx(float(want), abs=1e-12), measure


def test_the_tables_behind_the_figures():
    counts = qdapy.code_counts(FIXTURE)
    assert counts["n"].sum() == len(FIXTURE)
    assert list(counts["n"]) == sorted(counts["n"], reverse=True)
    wide = qdapy.code_matrix(FIXTURE, long=False)
    assert wide.to_numpy().sum() == len(FIXTURE)
    long = qdapy.code_matrix(FIXTURE, long=True)
    assert long["n"].sum() == len(FIXTURE)
    assert list(long.columns) == ["code", "document", "n"]


def test_code_matrix_falls_back_to_the_title_without_a_citekey():
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s2"], "title": ["Erste", "Zweite"],
        "code": ["A", "B"],
    })
    long = qdapy.code_matrix(frag)
    assert set(long["document"]) == {"Erste", "Zweite"}


def test_the_plotting_views_carry_their_caption_number():
    """Both drawing backends read the caption figure off ``attrs`` rather
    than off the result object, so a missing key would quietly become a
    wrong caption instead of an error."""
    points = qdapy.mds_points(FIXTURE, min_n=3)
    assert {"code", "dim1", "dim2"} <= set(points.columns)
    assert 0.0 <= points.attrs["goodness"] <= 1.0

    ca = qdapy.ca_points(FIXTURE)
    assert {"label", "dim1", "dim2", "kind"} <= set(ca.columns)
    assert set(ca["kind"]) == {"code", "document"}
    assert 0.0 < ca.attrs["inertia_shown"] <= 1.0
