"""The same figures drawn with seaborn.

Same arguments, same tables, same figures as :mod:`qdapy.gg` -- a different
grammar.  seaborn is the more idiomatic choice inside an existing matplotlib
workflow: every function returns a :class:`matplotlib.axes.Axes` (or a seaborn
grid), so the result can be placed in a subplot, restyled, or saved with the
rest of a figure.

Nothing here calls ``plt.show()``.  A library that draws to the screen by
itself cannot be used to build a figure.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from scipy.cluster import hierarchy

from . import frames, stats

__all__ = [
    "ca_map",
    "code_matrix",
    "dendrogram",
    "frequencies",
    "level_agreement",
    "mds",
    "saturation",
    "timeline",
]


def _axes(ax: Axes | None, **kwargs: Any) -> Axes:
    """The axes to draw on: the caller's, or a fresh figure."""
    if ax is not None:
        return ax
    return plt.subplots(**kwargs)[1]


def frequencies(
    fragments: pd.DataFrame,
    *,
    top: int | None = 25,
    fill: str = "#4c78a8",
    ax: Axes | None = None,
) -> Axes:
    """How often each code was assigned."""
    d = frames.code_counts(fragments, top=top)
    ax = _axes(ax)
    sns.barplot(data=d, y="code", x="n", color=fill, ax=ax)
    ax.set(xlabel="codings", ylabel="")
    sns.despine(ax=ax)
    return ax


def code_matrix(
    fragments: pd.DataFrame,
    *,
    doc_col: str = "citekey",
    annot: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Which codes occur in which document."""
    table = frames.code_matrix(fragments, doc_col=doc_col, long=False)
    ax = _axes(ax)
    sns.heatmap(table, cmap="Blues", annot=annot, fmt="d", cbar_kws={
        "label": "codings"}, ax=ax)
    ax.set(xlabel="", ylabel="")
    return ax


def timeline(history: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """How the number of codings grew, per coder."""
    d = frames.timeline(history)
    ax = _axes(ax)
    sns.lineplot(data=d, x="time", y="cumulative", hue="user",
                 drawstyle="steps-post", ax=ax)
    ax.set(xlabel="", ylabel="codings (cumulative)")
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("coder")
    sns.despine(ax=ax)
    return ax


def saturation(history: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """How many distinct codes existed after each successive coding."""
    d = frames.saturation(history)
    ax = _axes(ax)
    sns.lineplot(data=d, x="step", y="codes", ax=ax)
    ax.set(xlabel="codings", ylabel="distinct codes so far")
    sns.despine(ax=ax)
    return ax


def mds(
    fragments: pd.DataFrame,
    *,
    unit: str = "annotationKey",
    min_n: int = 3,
    ax: Axes | None = None,
) -> Axes:
    """A map of the codes by the segments they share."""
    d = stats.mds_points(fragments, unit=unit, min_n=min_n)
    ax = _axes(ax)
    sns.scatterplot(data=d, x="dim1", y="dim2", ax=ax)
    for _, row in d.iterrows():
        ax.annotate(row["code"], (row["dim1"], row["dim2"]),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=8)
    ax.set(xlabel="dimension 1", ylabel="dimension 2")
    ax.set_title(f"eigenvalue mass carried: {d.attrs['goodness']:.0%}",
                 fontsize=9, loc="left")
    sns.despine(ax=ax)
    return ax


def level_agreement(
    matrix: pd.DataFrame,
    *,
    max_level: int | None = None,
    measures: tuple[str, ...] = ("percent", "fleiss", "alpha"),
    ax: Axes | None = None,
) -> Axes:
    """Agreement against the number of code-system levels kept."""
    from . import agreement as ag

    d = ag.level_agreement(matrix, max_level=max_level)
    long = d.melt(id_vars="level", value_vars=list(measures),
                  var_name="measure", value_name="value")
    ax = _axes(ax)
    sns.lineplot(data=long, x="level", y="value", hue="measure", marker="o",
                 ax=ax)
    ax.set(xlabel="levels of the code system kept", ylabel="")
    ax.set_xticks(sorted(d["level"].unique()))
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("")
    sns.despine(ax=ax)
    return ax


def ca_map(
    fragments: pd.DataFrame,
    *,
    doc_col: str = "citekey",
    ax: Axes | None = None,
) -> Axes:
    """Codes and documents in one correspondence-analysis map."""
    d = stats.ca_points(fragments, doc_col=doc_col)
    ax = _axes(ax)
    sns.scatterplot(data=d, x="dim1", y="dim2", hue="kind", style="kind", ax=ax)
    for _, row in d.iterrows():
        ax.annotate(row["label"], (row["dim1"], row["dim2"]),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=8)
    ax.axhline(0, lw=0.5, color="grey")
    ax.axvline(0, lw=0.5, color="grey")
    ax.set(xlabel="dimension 1", ylabel="dimension 2")
    share = d.attrs["inertia_shown"]
    ax.set_title(f"inertia shown: {share:.0%}", fontsize=9, loc="left")
    sns.despine(ax=ax)
    return ax


def dendrogram(
    fragments: pd.DataFrame,
    *,
    unit: str = "annotationKey",
    min_n: int = 3,
    method: str = "average",
    ax: Axes | None = None,
) -> Axes:
    """The code clustering as a tree.

    The title carries the cophenetic correlation, so the number that says
    whether the tree may be read at all travels with the picture.
    """
    result = stats.cluster(fragments, unit=unit, min_n=min_n, method=method)
    ax = _axes(ax)
    hierarchy.dendrogram(result.linkage, labels=result.labels,
                         orientation="right", ax=ax)
    ax.set(xlabel="Jaccard distance")
    coph = result.cophenetic
    text = ("undefined (fewer than three codes)" if math.isnan(coph)
            else f"{coph:.2f}")
    ax.set_title(f"cophenetic correlation: {text}", fontsize=9, loc="left")
    sns.despine(ax=ax, left=True)
    return ax
