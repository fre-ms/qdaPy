"""The coding process over time: saturation as a number, and coder drift.

Both read the coding log, which zotQDA keeps and comparable tools do not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import pandas as pd

__all__ = ["code_drift", "codings", "new_codes", "saturation_ratio"]


def codings(
    history: pd.DataFrame,
    *,
    unit: str = "annotationKey",
    user: str = "user",
    action: str = "action",
    code: str = "code",
    ts: str = "ts",
) -> pd.DataFrame:
    """Reconstruct the current per-coder coding state from the log.

    zotQDA's *fragments* export is a last-state table: one row per annotation
    and code, with a single ``codedBy``.  Where two coders coded the same
    segment, that shape collapses -- the fragments file cannot tell you which
    coders agreed, only the last state.  Intercoder reliability therefore has
    to come from the *history* export, which keeps one row per coding event.

    This replays that log.  Every ``add`` and ``remove`` is applied in ``ts``
    order for each ``(annotationKey, user, code)`` triple, and the triple is
    kept only when its last event was an ``add``.  The result is the per-coder
    coding that actually stood at the end -- the same reconstruction qdaZ
    itself makes -- so it can be reshaped for agreement:

        >>> hist = qdapy.read_history(qdapy.example("zotqda-history-demo.csv"))
        >>> matrix = qdapy.units(qdapy.codings(hist), coder="user")
        >>> qdapy.agreement.agreement(matrix)  # doctest: +SKIP

    Parameters
    ----------
    history:
        The coding log, as read by :func:`qdapy.read_history`.
    unit, user, action, code, ts:
        Column names, should the log ever carry different ones.

    Returns
    -------
    pandas.DataFrame
        One row per surviving ``(annotationKey, user, code)`` triple, with
        columns ``annotationKey``, ``user`` and ``code`` -- plus ``citekey``
        and ``title`` when the log carries them, so the reconstruction stays a
        drop-in for the fragments the corpus tables expect.
    """
    for col in (unit, user, action, code, ts):
        if col not in history.columns:
            raise KeyError(f"history is missing the column {col!r}")

    ordered = history.sort_values(ts, kind="stable")
    last = ordered.drop_duplicates(subset=[unit, user, code], keep="last")
    kept = last[last[action].astype(str) == "add"]

    carry = [c for c in ("citekey", "title") if c in history.columns]
    return kept[[unit, user, code, *carry]].reset_index(drop=True)


def new_codes(history: pd.DataFrame, *, doc_col: str = "citekey") -> pd.DataFrame:
    """How many codes appeared for the first time in each document.

    Documents are ordered by when they were first coded, which is the order
    the saturation argument is about.
    """
    added = history[history["action"] == "add"].sort_values("ts")
    docs: list[str] = []
    counts: list[int] = []
    seen: set[str] = set()
    for doc, code in zip(added[doc_col].astype(str), added["code"].astype(str),
                         strict=True):
        if doc not in docs:
            docs.append(doc)
            counts.append(0)
        if code not in seen:
            seen.add(code)
            counts[docs.index(doc)] += 1
    out = pd.DataFrame({"position": range(1, len(docs) + 1), "document": docs,
                        "new_codes": counts})
    out["cumulative"] = out["new_codes"].cumsum()
    return out


class Saturation(TypedDict):
    """What :func:`saturation_ratio` reports.

    ``notation`` is the string a methods section quotes ("6+2"), or ``None``
    when saturation was never reached -- in which case ``reason`` says why in
    words rather than leaving the reader to infer it from a missing number.
    """

    base_size: int
    run_length: int
    threshold: float
    base_codes: float
    documents: int
    runs: pd.DataFrame
    saturated_at: int | None
    notation: str | None
    reason: str


def saturation_ratio(
    counts: Sequence[float],
    *,
    base_size: int = 4,
    run_length: int = 2,
    threshold: float = 0.05,
) -> Saturation:
    """Saturation as something you can put in a methods section.

    A curve shows a trend; it does not answer "how many documents were
    enough".  Guest, Namey and Chen (2020) doi:10.1371/journal.pone.0232076
    operationalised the question: a base of documents whose codes count as
    what is already known (they recommend four), a run of consecutive later
    documents inspected for new codes, and the share of new information that
    still counts as saturated.  The result reads ``"6+2"``: saturation
    declared at document six, confirmed over a run of two.

    This is *code* saturation and nothing else.  Hennink, Kaiser and Marconi
    (2017) doi:10.1177/1049732316665344 distinguish it from meaning
    saturation, which no algorithm sees; Braun and Clarke (2019)
    doi:10.1080/2159676X.2019.1704846 reject saturation as a criterion for
    reflexive thematic analysis altogether.  Report which conception you
    mean.
    """
    if isinstance(counts, pd.DataFrame):
        counts = counts["new_codes"]
    values = [max(0.0, float(c or 0)) for c in list(counts)]
    base = max(1, int(base_size))
    run = max(1, int(run_length))
    n = len(values)
    base_codes = sum(values[:base])

    runs = []
    reached = None
    if n >= base + run and base_codes > 0:
        for start in range(base, n - run + 1):
            new_in_run = sum(values[start:start + run])
            ratio = new_in_run / base_codes
            ok = ratio <= threshold
            runs.append({"from": start + 1, "to": start + run,
                         "new_codes": new_in_run, "ratio": ratio, "ok": ok})
            if ok and reached is None:
                reached = start
    return {
        "base_size": base, "run_length": run, "threshold": threshold,
        "base_codes": base_codes, "documents": n,
        "runs": pd.DataFrame(runs, columns=["from", "to", "new_codes",
                                            "ratio", "ok"]),
        "saturated_at": reached,
        "notation": None if reached is None else f"{reached}+{run}",
        "reason": ("too few documents for one full run" if n < base + run
                   else "no codes in the base set" if base_codes == 0 else ""),
    }


def code_drift(history: pd.DataFrame, *, windows: int = 4) -> pd.DataFrame:
    """Did a coder's behaviour shift while the project ran?

    The coding log records who coded what and when, so this question can be
    asked at all -- most tools keep no such trail.  Reported per window as
    the total variation distance between that coder's code distribution and
    their first window: nought means they are coding as they started, one
    that the two windows share no code.

    Windows hold equal numbers of events rather than equal spans of time,
    because a coder who worked in bursts would otherwise get empty ones.

    It is a description, not a test.  A large distance can mean drift, or
    that the later material was simply about something else.
    """
    w = max(2, int(windows))
    d = history[(history["action"] == "add")
                & history["code"].astype(str).str.len().gt(0)
                & history["user"].astype(str).str.len().gt(0)
                & history["ts"].astype(str).str.len().gt(0)].sort_values("ts")
    rows = []
    for who in sorted(d["user"].astype(str).unique()):
        mine = d[d["user"].astype(str) == who]
        per = -(-len(mine) // w)          # ceiling division
        chunks = [mine.iloc[i:i + per] for i in range(0, len(mine), per)]
        shares = [c["code"].astype(str).value_counts(normalize=True)
                  for c in chunks]
        first = shares[0]
        for k, (chunk, share) in enumerate(zip(chunks, shares, strict=True), 1):
            keys = set(first.index) | set(share.index)
            tv = sum(abs(first.get(c, 0.0) - share.get(c, 0.0)) for c in keys) / 2
            rows.append({"coder": who, "window": k, "n": len(chunk),
                         "codes": len(share),
                         "from": str(chunk["ts"].iloc[0]),
                         "to": str(chunk["ts"].iloc[-1]),
                         "distance": tv})
    return pd.DataFrame(rows, columns=["coder", "window", "n", "codes",
                                       "from", "to", "distance"])
