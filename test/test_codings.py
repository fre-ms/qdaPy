"""``codings()``: replay the coding log to the state that actually stood.

zotQDA's fragments export is last-state and loses per-coder agreement; the
per-coder truth lives in the history.  ``codings()`` reconstructs it, so
``units(codings(hist), coder="user")`` is the reliability entry point.
"""

from __future__ import annotations

import pandas as pd
import pytest

import qdapy


def log(rows):
    """A minimal history frame from ``(ts, user, action, code, key)`` tuples."""
    return pd.DataFrame(
        rows, columns=["ts", "user", "action", "code", "annotationKey"]
    )


def test_last_add_survives_a_later_remove():
    h = log([
        ("2026-01-01T09:00:00Z", "ann", "add", "A", "K1"),
        ("2026-01-02T09:00:00Z", "ann", "remove", "A", "K1"),
    ])
    assert qdapy.codings(h).empty


def test_remove_then_readd_is_kept():
    h = log([
        ("2026-01-01T09:00:00Z", "ann", "add", "A", "K1"),
        ("2026-01-02T09:00:00Z", "ann", "remove", "A", "K1"),
        ("2026-01-03T09:00:00Z", "ann", "add", "A", "K1"),
    ])
    out = qdapy.codings(h)
    assert len(out) == 1
    assert out.iloc[0]["code"] == "A"


def test_events_are_applied_in_ts_order_not_row_order():
    # the remove is the last event by ts although it appears first in the frame
    h = log([
        ("2026-01-09T09:00:00Z", "ann", "remove", "A", "K1"),
        ("2026-01-01T09:00:00Z", "ann", "add", "A", "K1"),
    ])
    assert qdapy.codings(h).empty


def test_triples_are_independent():
    h = log([
        ("2026-01-01T09:00:00Z", "ann", "add", "A", "K1"),
        ("2026-01-01T09:00:00Z", "bob", "add", "A", "K1"),
        ("2026-01-02T09:00:00Z", "ann", "remove", "A", "K1"),
    ])
    out = qdapy.codings(h)
    assert list(out["user"]) == ["bob"]


def test_carries_citekey_and_title_when_present():
    h = log([("2026-01-01T09:00:00Z", "ann", "add", "A", "K1")])
    h["citekey"] = "d1"
    h["title"] = "Interview d1"
    out = qdapy.codings(h)
    assert list(out.columns) == ["annotationKey", "user", "code",
                                 "citekey", "title"]
    assert out.iloc[0]["citekey"] == "d1"


def test_missing_column_raises():
    with pytest.raises(KeyError):
        qdapy.codings(pd.DataFrame({"ts": [], "user": [], "action": []}))


def test_reconstructs_demo_reliability_from_history():
    hist = qdapy.read_history(qdapy.example("easyqda-history-demo.csv"))
    codings = qdapy.codings(hist)
    assert set(codings.columns) >= {"annotationKey", "user", "code"}
    assert sorted(codings["user"].unique()) == ["ann", "bob"]
    a = qdapy.agreement.agreement(qdapy.units(codings, coder="user"))
    assert float(a["alpha"].iloc[0]) == pytest.approx(0.6716, abs=1e-4)
