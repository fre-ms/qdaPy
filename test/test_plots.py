"""The three backends must draw the same figures from the same tables.

A plotting test cannot check that a figure looks right.  It can check the two
things that actually break: that the figure is built at all, and that the
numbers reaching it are the numbers the table holds -- which is what would
silently differ if one backend started preparing its own data.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import plotnine as p9
import pytest
from matplotlib.axes import Axes

import qdapy

matplotlib.use("Agg")   # no display in a test run

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIXTURE = pd.read_csv(HERE / "stats-fixture.csv", dtype=str)

HISTORY = pd.DataFrame({
    "ts": ["2026-01-01T09:00:00.000Z", "2026-01-01T09:05:00.000Z",
           "2026-01-01T09:10:00.000Z", "2026-01-01T09:15:00.000Z",
           "2026-01-01T09:20:00.000Z"],
    "user": ["ann", "bob", "ann", "ann", "bob"],
    "action": ["add", "add", "add", "remove", "add"],
    "code": ["A", "A", "B", "B", "C"],
})

UNITS = pd.DataFrame({
    "ann": ["A/x", "A/y", "B/x", "B/y"],
    "bob": ["A/y", "A/y", "B/x", "B/x"],
}, dtype=object)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --- the tables the figures are built from ----------------------------


def test_the_timeline_counts_only_additions():
    d = qdapy.timeline(HISTORY)
    assert len(d) == 4                      # the remove does not count
    assert list(d[d["user"] == "ann"]["cumulative"]) == [1, 2]
    assert list(d[d["user"] == "bob"]["cumulative"]) == [1, 2]


def test_the_timeline_of_a_log_without_additions_is_empty_not_broken():
    only_removes = HISTORY[HISTORY["action"] == "remove"]
    d = qdapy.timeline(only_removes)
    assert d.empty
    assert list(d.columns) == ["time", "user", "cumulative"]


def test_the_timeline_needs_its_columns():
    with pytest.raises(KeyError, match="action"):
        qdapy.timeline(pd.DataFrame({"ts": ["2026-01-01T00:00:00Z"],
                                    "user": ["ann"]}))


def test_the_saturation_curve_counts_distinct_codes_and_never_falls():
    d = qdapy.saturation(HISTORY)
    assert list(d["codes"]) == [1, 1, 2, 3]   # A, A, B, C
    assert list(d["step"]) == [1, 2, 3, 4]
    assert all(b >= a for a, b in zip(d["codes"], d["codes"][1:], strict=False))


# --- plotnine ---------------------------------------------------------


def test_plotnine_frequencies_carries_the_counts():
    plot = qdapy.gg.frequencies(FIXTURE)
    assert isinstance(plot, p9.ggplot)
    counts = qdapy.code_counts(FIXTURE)
    assert plot.data["n"].sum() == counts["n"].sum()
    assert len(plot.data) == len(counts)


def test_plotnine_frequencies_honours_top():
    plot = qdapy.gg.frequencies(FIXTURE, top=3)
    assert len(plot.data) == 3


def test_plotnine_orders_the_bars_by_frequency():
    plot = qdapy.gg.frequencies(FIXTURE)
    # the categorical order is what turns a bar chart into a ranking
    order = list(plot.data["code"].cat.categories)
    counts = qdapy.code_counts(FIXTURE).set_index("code")["n"]
    assert [counts[c] for c in order] == sorted(counts.tolist())


@pytest.mark.parametrize(
    ("name", "argument"),
    [
        ("frequencies", "fragments"),
        ("code_matrix", "fragments"),
        ("timeline", "history"),
        ("saturation", "history"),
        ("mds", "fragments"),
        ("ca_map", "fragments"),
        ("level_agreement", "units"),
    ],
)
def test_every_plotnine_figure_builds(name, argument):
    data = {"fragments": FIXTURE, "history": HISTORY, "units": UNITS}[argument]
    kwargs = {"min_n": 3} if name == "mds" else {}
    plot = getattr(qdapy.gg, name)(data, **kwargs)
    assert isinstance(plot, p9.ggplot)
    # building the figure is where a bad mapping shows up
    plot._build()


# --- seaborn ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "argument"),
    [
        ("frequencies", "fragments"),
        ("code_matrix", "fragments"),
        ("timeline", "history"),
        ("saturation", "history"),
        ("mds", "fragments"),
        ("ca_map", "fragments"),
        ("dendrogram", "fragments"),
        ("level_agreement", "units"),
    ],
)
def test_every_seaborn_figure_returns_axes(name, argument):
    data = {"fragments": FIXTURE, "history": HISTORY, "units": UNITS}[argument]
    kwargs = {"min_n": 3} if name in {"mds", "dendrogram"} else {}
    ax = getattr(qdapy.sns, name)(data, **kwargs)
    assert isinstance(ax, Axes)


def test_seaborn_draws_into_the_axes_it_is_given():
    fig, (left, right) = plt.subplots(1, 2)
    a = qdapy.sns.frequencies(FIXTURE, ax=left)
    b = qdapy.sns.saturation(HISTORY, ax=right)
    assert a is left
    assert b is right
    assert len(fig.axes) == 2


def test_seaborn_frequencies_draws_one_bar_per_code():
    ax = qdapy.sns.frequencies(FIXTURE)
    assert len(ax.patches) == len(qdapy.code_counts(FIXTURE))


def test_the_dendrogram_states_the_cophenetic_correlation():
    ax = qdapy.sns.dendrogram(FIXTURE, min_n=3)
    coph = qdapy.cluster(FIXTURE, min_n=3).cophenetic
    assert f"{coph:.2f}" in ax.get_title(loc="left")


def test_the_dendrogram_says_so_when_the_correlation_is_undefined():
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s1", "s2", "s3"],
        "code": ["A", "B", "A", "B"],
    })
    ax = qdapy.sns.dendrogram(frag, min_n=1)
    assert "undefined" in ax.get_title(loc="left")


def test_the_code_map_states_how_much_it_shows():
    ax = qdapy.sns.mds(FIXTURE, min_n=3)
    share = qdapy.mds(FIXTURE, min_n=3).goodness[0]
    assert f"{share:.0%}" in ax.get_title(loc="left")


def test_the_correspondence_map_states_the_inertia_shown():
    ax = qdapy.sns.ca_map(FIXTURE)
    share = sum(qdapy.ca(FIXTURE).inertia_share[:2])
    assert f"{share:.0%}" in ax.get_title(loc="left")
    # and it must not claim the whole table
    assert share < 1


def test_no_backend_opens_a_window_by_itself():
    before = plt.get_fignums()
    qdapy.gg.frequencies(FIXTURE)
    assert plt.get_fignums() == before


# --- both backends agree ----------------------------------------------


def test_both_backends_show_the_same_number_of_codes():
    plot = qdapy.gg.frequencies(FIXTURE, top=None)
    ax = qdapy.sns.frequencies(FIXTURE, top=None)
    assert len(plot.data) == len(ax.patches)


def test_both_backends_draw_the_same_level_curve():
    plot = qdapy.gg.level_agreement(UNITS)
    ax = qdapy.sns.level_agreement(UNITS)
    levels = sorted(plot.data["level"].unique())
    assert [int(t) for t in ax.get_xticks()] == levels


# --- the plugin's own charts ------------------------------------------


def test_a_spec_states_its_provenance(tmp_path):
    path = tmp_path / "chart.json"
    path.write_text(json.dumps({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "bar",
        "usermeta": {"contract": "easyqda-exchange", "version": 1,
                     "producer": "qdaZ", "analysis": "frequencies"},
    }), encoding="utf-8")
    spec = qdapy.vega.read_spec(path)
    assert spec["usermeta"]["producer"] == qdapy.vega.PRODUCER
    assert spec["usermeta"]["analysis"] == "frequencies"


def test_a_spec_from_a_newer_exchange_version_warns(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"mark": "bar", "usermeta": {"version": 99}}),
                    encoding="utf-8")
    with pytest.warns(UserWarning, match="exchange version 99"):
        qdapy.vega.read_spec(path)


def test_a_spec_without_metadata_is_still_read(tmp_path):
    path = tmp_path / "plain.json"
    path.write_text(json.dumps({"mark": "point"}), encoding="utf-8")
    assert qdapy.vega.read_spec(path)["mark"] == "point"


def test_rendering_inlines_the_data():
    altair = pytest.importorskip("altair")
    spec = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": "bar",
            "encoding": {"x": {"field": "code", "type": "nominal"},
                         "y": {"field": "n", "type": "quantitative"}}}
    counts = qdapy.code_counts(FIXTURE)
    chart = qdapy.vega.render(spec, counts)
    assert isinstance(chart, altair.Chart)
    spec_out = chart.to_dict()
    rows = next(iter(spec_out["datasets"].values()))
    assert len(rows) == len(counts)
    assert {r["code"] for r in rows} == set(counts["code"])
