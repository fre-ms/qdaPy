"""The plugin's figures in the grammar of graphics, drawn with plotnine.

This is the backend closest to the R twin: plotnine is a ggplot2 dialect, so a
figure built here and one built in qdaR are the same specification written
twice.  Every function takes the same arguments as its counterpart in
:mod:`qdapy.sns` and draws from the same table in :mod:`qdapy.frames`, so
switching backends changes the appearance and nothing else.
"""

from __future__ import annotations

import pandas as pd
import plotnine as p9

from . import frames, stats

__all__ = [
    "ca_map",
    "code_matrix",
    "frequencies",
    "level_agreement",
    "mds",
    "saturation",
    "timeline",
]


def frequencies(
    fragments: pd.DataFrame,
    *,
    top: int | None = 25,
    fill: str = "#4c78a8",
) -> p9.ggplot:
    """How often each code was assigned."""
    d = frames.code_counts(fragments, top=top)
    order = list(d.sort_values("n")["code"])
    d = d.assign(code=pd.Categorical(d["code"], categories=order, ordered=True))
    return (
        p9.ggplot(d, p9.aes(x="code", y="n"))
        + p9.geom_col(fill=fill)
        + p9.coord_flip()
        + p9.labs(x="", y="codings")
        + p9.theme_minimal()
    )


def code_matrix(
    fragments: pd.DataFrame, *, doc_col: str = "citekey"
) -> p9.ggplot:
    """Which codes occur in which document."""
    d = frames.code_matrix(fragments, doc_col=doc_col, long=True)
    return (
        p9.ggplot(d, p9.aes(x="document", y="code", fill="n"))
        + p9.geom_tile()
        + p9.scale_fill_gradient(low="#f0f4f8", high="#2b5d8a")
        + p9.labs(x="", y="", fill="codings")
        + p9.theme_minimal()
        + p9.theme(axis_text_x=p9.element_text(angle=45, ha="right"))
    )


def timeline(history: pd.DataFrame) -> p9.ggplot:
    """How the number of codings grew, per coder."""
    d = frames.timeline(history)
    return (
        p9.ggplot(d, p9.aes(x="time", y="cumulative", colour="user"))
        + p9.geom_step()
        + p9.labs(x="", y="codings (cumulative)", colour="coder")
        + p9.theme_minimal()
    )


def saturation(history: pd.DataFrame) -> p9.ggplot:
    """How many distinct codes existed after each successive coding."""
    d = frames.saturation(history)
    return (
        p9.ggplot(d, p9.aes(x="step", y="codes"))
        + p9.geom_line()
        + p9.labs(x="codings", y="distinct codes so far")
        + p9.theme_minimal()
    )


def mds(
    fragments: pd.DataFrame, *, unit: str = "annotationKey", min_n: int = 3
) -> p9.ggplot:
    """A map of the codes by the segments they share."""
    d = stats.mds_points(fragments, unit=unit, min_n=min_n)
    share = d.attrs["goodness"]
    return (
        p9.ggplot(d, p9.aes(x="dim1", y="dim2"))
        + p9.geom_point()
        + p9.geom_text(p9.aes(label="code"), nudge_y=0.03, size=8)
        + p9.labs(x="dimension 1", y="dimension 2",
                  caption=f"eigenvalue mass carried: {share:.0%}")
        + p9.theme_minimal()
    )


def level_agreement(
    matrix: pd.DataFrame,
    *,
    max_level: int | None = None,
    measures: tuple[str, ...] = ("percent", "fleiss", "alpha"),
) -> p9.ggplot:
    """Agreement against the number of code-system levels kept."""
    from . import agreement as ag

    d = ag.level_agreement(matrix, max_level=max_level)
    long = d.melt(id_vars="level", value_vars=list(measures),
                  var_name="measure", value_name="value")
    return (
        p9.ggplot(long, p9.aes(x="level", y="value", colour="measure"))
        + p9.geom_line()
        + p9.geom_point()
        + p9.scale_x_continuous(breaks=sorted(d["level"].unique()))
        + p9.labs(x="levels of the code system kept", y="", colour="")
        + p9.theme_minimal()
    )


def ca_map(
    fragments: pd.DataFrame, *, doc_col: str = "citekey"
) -> p9.ggplot:
    """Codes and documents in one correspondence-analysis map.

    The caption states how much of the table's inertia the two dimensions
    carry, because a map that shows a fifth of the structure looks exactly
    like one that shows all of it.
    """
    d = stats.ca_points(fragments, doc_col=doc_col)
    share = d.attrs["inertia_shown"]
    return (
        p9.ggplot(d, p9.aes(x="dim1", y="dim2", colour="kind"))
        + p9.geom_point()
        + p9.geom_text(p9.aes(label="label"), nudge_y=0.03, size=8)
        + p9.labs(x="dimension 1", y="dimension 2", colour="",
                  caption=f"inertia shown: {share:.0%}")
        + p9.theme_minimal()
    )
