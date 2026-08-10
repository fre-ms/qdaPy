"""The demo export ships in qdaPy and qdaR and must be the same study.

``zotqda-fragments-demo.csv`` and ``zotqda-history-demo.csv`` are written
by ``scripts/make_demo_export.py`` into both packages at once.  These
tests freeze the study's shape and its headline coefficients, and hold
the two shipped copies to byte identity — the same discipline the
frozen qdaZ references follow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import qdapy


def test_demo_fragments_shape():
    frag = qdapy.read_fragments(qdapy.example("zotqda-fragments-demo.csv"))
    assert len(frag) == 502
    assert frag["code"].nunique() == 9
    assert frag["citekey"].nunique() == 8
    assert sorted(frag["codedBy"].unique()) == ["ann", "bob"]
    assert (frag["positionKind"] == "text").all()


def test_demo_agreement_frozen():
    frag = qdapy.read_fragments(qdapy.example("zotqda-fragments-demo.csv"))
    a = qdapy.agreement.agreement(qdapy.units(frag))
    assert float(a["alpha"].iloc[0]) == pytest.approx(0.7266, abs=1e-4)
    assert float(a["ac1"].iloc[0]) == pytest.approx(0.7299, abs=1e-4)
    assert int(a["multi_set_aside"].iloc[0]) > 0


def test_demo_history_reads():
    hist = qdapy.read_history(qdapy.example("zotqda-history-demo.csv"))
    assert set(hist["action"]) == {"add", "remove"}
    assert (hist["action"] == "remove").sum() == 6


@pytest.mark.parametrize("name", ["zotqda-fragments-demo.csv",
                                  "zotqda-history-demo.csv"])
def test_demo_identical_in_qdar(name):
    theirs = Path(__file__).parents[2] / "qdaR" / "inst" / "extdata" / name
    if not theirs.exists():
        pytest.skip("qdaR sources not present")
    ours = Path(qdapy.example(name))
    assert ours.read_bytes() == theirs.read_bytes()
