"""The tables behind the figures.

Every plotting backend draws from these, so the three backends cannot drift
apart in what they show -- only in how they show it.  They are useful on their
own: the numbers behind a figure are what goes into a paper.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["code_counts", "code_matrix", "saturation", "timeline"]


def code_counts(fragments: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    """Counts per code, most frequent first.

    Parameters
    ----------
    top:
        Keep only the most frequent codes.  ``None`` keeps all.
    """
    codes = fragments.loc[fragments["code"].astype(str).str.len() > 0, "code"]
    counts = codes.astype(str).value_counts()
    out = pd.DataFrame({"code": counts.index.astype(str),
                        "n": counts.to_numpy(dtype=int)})
    if top is not None:
        out = out.head(top).reset_index(drop=True)
    return out


def code_matrix(
    fragments: pd.DataFrame,
    *,
    doc_col: str = "citekey",
    long: bool = True,
) -> pd.DataFrame:
    """The code-by-document table.

    Documents are identified by ``citekey`` when present, otherwise by title.
    ``long=True`` returns one row per cell, which is what the plotting
    backends want; ``long=False`` returns the table itself.
    """
    if doc_col not in fragments.columns:
        doc_col = "title"
    coded = fragments[fragments["code"].astype(str).str.len() > 0]
    table = pd.crosstab(coded["code"].astype(str), coded[doc_col].astype(str))
    table.index.name = "code"
    table.columns.name = "document"
    if not long:
        return table
    out = table.stack().reset_index()
    out.columns = ["code", "document", "n"]
    return out


def timeline(history: pd.DataFrame) -> pd.DataFrame:
    """Cumulative codings per coder over time.

    Only ``add`` events count: a coding that was later removed did happen, but
    it is not part of the material at the end.
    """
    for col in ("ts", "user", "action"):
        if col not in history.columns:
            raise KeyError(f"history is missing the column {col!r}")
    added = history[history["action"] == "add"].copy()
    if added.empty:
        return pd.DataFrame({"time": pd.Series(dtype="datetime64[ns, UTC]"),
                             "user": pd.Series(dtype=str),
                             "cumulative": pd.Series(dtype=int)})
    added["time"] = pd.to_datetime(added["ts"], format="ISO8601", utc=True)
    added = added.sort_values("time")
    added["cumulative"] = added.groupby("user").cumcount() + 1
    return added[["time", "user", "cumulative"]].reset_index(drop=True)


def saturation(history: pd.DataFrame) -> pd.DataFrame:
    """How many distinct codes existed after each successive coding.

    The curve flattens when a code system stops growing, which is the
    empirical form of the saturation argument -- and a curve that never
    flattens is worth reporting as such.
    """
    added = history[history["action"] == "add"].sort_values("ts")
    seen: set[str] = set()
    counts = []
    for code in added["code"].astype(str):
        seen.add(code)
        counts.append(len(seen))
    return pd.DataFrame({"step": range(1, len(counts) + 1), "codes": counts})
