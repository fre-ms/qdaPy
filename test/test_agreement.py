"""The coefficients, against values a person can check by hand.

The same examples are in the R twin's test suite, with the same arithmetic
written out in the comments -- so a disagreement between the two packages shows
up as a disagreement with a hand computation, not merely with each other.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import qdapy
from qdapy.agreement import agreement


def matrix(**columns) -> pd.DataFrame:
    m = pd.DataFrame(columns, dtype=object)
    m.attrs["multi"] = 0
    return m


def test_units_are_built_per_segment_and_coder():
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s1", "s2", "s2", "s3"],
        "codedBy": ["ann", "bob", "ann", "bob", "ann"],
        "code": ["A", "A", "B", "A", "A"],
    })
    u = qdapy.units(frag)
    assert u.shape == (3, 2)
    assert list(u.columns) == ["ann", "bob"]
    assert list(u.loc["s1"]) == ["A", "A"]
    assert u.loc["s3", "bob"] is None      # bob never rated s3
    assert u.attrs["multi"] == 0


def test_a_segment_one_coder_coded_twice_is_set_aside_and_counted():
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s1", "s1", "s2", "s2"],
        "codedBy": ["ann", "ann", "bob", "ann", "bob"],
        "code": ["A", "B", "A", "B", "B"],
    })
    u = qdapy.units(frag)
    assert u.loc["s1", "ann"] is None
    assert u.loc["s1", "bob"] == "A"
    assert u.attrs["multi"] == 1
    # and it is reported rather than swallowed
    assert agreement(u).iloc[0]["multi_set_aside"] == 1


def test_the_binary_view_keeps_multiply_coded_segments():
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s1", "s1", "s2", "s2"],
        "codedBy": ["ann", "ann", "bob", "ann", "bob"],
        "code": ["A", "B", "A", "B", "B"],
    })
    b = qdapy.units_binary(frag, "A")
    assert list(b.loc["s1"]) == ["yes", "yes"]
    assert list(b.loc["s2"]) == ["no", "no"]
    assert qdapy.percent_agreement(b) == 1


def test_uncoded_segments_become_their_own_category():
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s1"],
        "codedBy": ["ann", "bob"],
        "code": ["A", "A"],
    })
    unc = pd.DataFrame({"annotationKey": ["s2"]})
    u = qdapy.units(frag, uncoded=unc)
    assert len(u) == 2
    assert list(u.loc["s2"]) == ["(no code)", "(no code)"]
    # agreement about irrelevance is agreement
    assert qdapy.percent_agreement(u) == 1
    # without the uncoded export that segment is simply absent
    assert len(qdapy.units(frag)) == 1


def test_cohens_kappa_matches_the_hand_computation():
    u = matrix(ann=["A", "B", "A", "B"], bob=["A", "B", "B", "B"])
    # po = 3/4, pe = .5*.25 + .5*.75 = .5
    assert qdapy.percent_agreement(u) == 0.75
    assert qdapy.kappa(u) == pytest.approx(0.5)


def test_cohens_kappa_refuses_three_coders_instead_of_using_the_first_two():
    u = matrix(ann=["A", "B"], bob=["A", "B"], cat=["A", "A"])
    with pytest.raises(ValueError, match="two coders"):
        qdapy.kappa(u)


def test_fleiss_kappa_matches_the_hand_computation():
    u = matrix(ann=["A", "B", "A"], bob=["A", "B", "B"], cat=["A", "B", "A"])
    # pa = 7/9, pe = (5/9)^2 + (4/9)^2 = 41/81  ->  22/40
    assert qdapy.fleiss(u) == pytest.approx(0.55)


def test_krippendorff_alpha_matches_the_hand_computation():
    u = matrix(ann=["A", "B", "A", "B"], bob=["A", "B", "B", "B"])
    # d_obs = 2, d_exp = 30/7  ->  1 - 14/30
    assert qdapy.alpha(u) == pytest.approx(8 / 15)


def test_alpha_tolerates_a_coder_who_skipped_a_unit():
    u = matrix(ann=["A", "B", "A", None], bob=["A", "B", "B", "A"])
    assert not math.isnan(qdapy.alpha(u))
    # the incomplete unit contributes nothing
    assert qdapy.alpha(u) == pytest.approx(qdapy.alpha(u.iloc[:3]))


def test_ac1_holds_up_where_kappa_collapses_on_skewed_marginals():
    u = matrix(ann=["A", "A", "A", "B"], bob=["A", "A", "A", "A"])
    assert qdapy.percent_agreement(u) == 0.75
    assert qdapy.kappa(u) == pytest.approx(0)     # marginal chance eats it all
    assert qdapy.ac1(u) == pytest.approx(0.68)    # (0.75 - 0.21875) / 0.78125


def test_brennan_uses_uniform_chance():
    u = matrix(ann=["A", "A", "A", "B"], bob=["A", "A", "A", "A"])
    assert qdapy.brennan(u) == pytest.approx(0.5)            # (0.75-0.5)/0.5
    assert qdapy.brennan(u, q=4) == pytest.approx(2 / 3)     # (0.75-0.25)/0.75


def test_undefined_measures_say_nan_instead_of_inventing_a_number():
    same = matrix(ann=["A", "A"], bob=["A", "A"])
    assert math.isnan(qdapy.kappa(same))     # pe = 1
    assert math.isnan(qdapy.fleiss(same))    # a single category
    assert math.isnan(qdapy.ac1(same))
    empty = matrix(ann=[], bob=[])
    assert math.isnan(qdapy.percent_agreement(empty))


def test_agreement_reports_every_measure_side_by_side():
    u = matrix(ann=["A", "B", "A", "B"], bob=["A", "B", "B", "B"])
    row = agreement(u).iloc[0]
    assert row["units"] == 4
    assert row["coders"] == 2
    assert row["categories"] == 2
    assert row["cohen"] == pytest.approx(0.5)
    # Cohen is not defined for three coders and must not be faked
    u3 = matrix(ann=["A", "B", "A", "B"], bob=["A", "B", "B", "B"],
                cat=["A", "B", "A", "B"])
    row3 = agreement(u3).iloc[0]
    assert math.isnan(row3["cohen"])
    assert math.isnan(row3["brennan"])
    assert not math.isnan(row3["fleiss"])


@pytest.mark.parametrize(
    ("path", "level", "expected"),
    [
        ("A/b/c", 2, "A/b"),
        ("A/b/c", 9, "A/b/c"),
        ("A/b/c", None, "A/b/c"),
        ("A/b/c", 0, "A/b/c"),
        ("A", 1, "A"),
    ],
)
def test_paths_flatten_to_a_level(path, level, expected):
    assert qdapy.flatten_path(path, level) == expected


def test_flatten_path_takes_a_sequence_too():
    assert qdapy.flatten_path(["A/b", "C"], 1) == ["A", "C"]


def test_agreement_improves_towards_the_top_of_the_code_system():
    u = matrix(ann=["A/x", "A/y", "B/x"], bob=["A/y", "A/y", "B/x"])
    lv = qdapy.level_agreement(u)
    assert list(lv["level"]) == [1, 2]
    assert lv.iloc[0]["percent"] == 1              # both call it A, A, B
    assert lv.iloc[1]["percent"] == pytest.approx(2 / 3)
    assert lv.iloc[0]["categories"] == 2
    assert lv.iloc[1]["categories"] == 3


def test_the_level_curve_honours_max_level():
    u = matrix(ann=["A/x/1", "A/y/1"], bob=["A/x/2", "A/y/1"])
    assert len(qdapy.level_agreement(u)) == 3
    assert len(qdapy.level_agreement(u, max_level=2)) == 2


def test_flattening_keeps_missing_ratings_missing():
    u = matrix(ann=["A/x", None], bob=["A/y", "B/x"])
    lv = qdapy.level_agreement(u)
    # the unit only one coder rated stays uncomparable at every level
    assert set(lv["units"]) == {1}


def test_the_confusion_table_names_the_pairs_that_cost_the_agreement():
    u = matrix(ann=["A", "B", "A", "B"], bob=["A", "B", "B", "B"])
    cf = qdapy.confusion(u)
    assert list(cf.columns) == ["ann", "bob", "n"]
    assert cf["n"].sum() == 4
    dis = qdapy.confusion(u, only_disagreements=True)
    assert len(dis) == 1
    assert dis.iloc[0]["ann"] == "A"
    assert dis.iloc[0]["bob"] == "B"
    assert dis.iloc[0]["n"] == 1


def test_the_confusion_table_is_for_two_coders():
    u = matrix(ann=["A"], bob=["A"], cat=["A"])
    with pytest.raises(ValueError, match="two coders"):
        qdapy.confusion(u)


def test_units_can_be_keyed_on_the_code_identity_instead_of_the_path():
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s1"],
        "codedBy": ["ann", "bob"],
        "code": ["Belastung", "Stress"],   # renamed between the two exports
        "codeId": ["cX1", "cX1"],
    })
    assert qdapy.percent_agreement(qdapy.units(frag)) == 0
    assert qdapy.percent_agreement(qdapy.units(frag, value="codeId")) == 1


def test_units_can_flatten_while_building():
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s1"],
        "codedBy": ["ann", "bob"],
        "code": ["A/x", "A/y"],
    })
    assert qdapy.percent_agreement(qdapy.units(frag)) == 0
    assert qdapy.percent_agreement(qdapy.units(frag, level=1)) == 1


def test_units_complains_about_a_missing_column():
    with pytest.raises(KeyError, match="codedBy"):
        qdapy.units(pd.DataFrame({"annotationKey": ["s1"], "code": ["A"]}))
