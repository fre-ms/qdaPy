"""Did the coders mark the same stretches of text at all?

Every coefficient in :mod:`qdapy.agreement` assumes the segments already
line up and only asks whether the categories match.  That assumption does a
lot of work.  Established QDA software settles the prior question with one
overlap threshold, yes or no, which discards exactly the information about
how the boundaries differ.

Two families are here, answering different questions.  Krippendorff's
unitizing alpha is chance-corrected and handles a continuum of coded units
and gaps.  WindowDiff and Pk come from text segmentation, are not
chance-corrected, and are comparable with that literature.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import TypedDict

import pandas as pd

__all__ = ["AlphaResult", "Segment", "pk", "segments", "unitizing_alpha",
           "window_diff"]


class Segment(TypedDict):
    """One coded stretch on the character continuum of a document.

    A ``TypedDict`` rather than a bare ``dict`` because this package ships
    ``py.typed``: the shape is part of what callers are promised, and a
    plain mapping would tell their type checker nothing.  Plain dict
    literals still satisfy it, so nothing changes at a call site.
    """

    start: float
    end: float
    value: str


class AlphaResult(TypedDict):
    """What :func:`unitizing_alpha` reports.

    ``Do`` and ``De`` are named after the equations rather than after Python
    convention: a reader checking the implementation against Krippendorff
    should find the symbols the paper uses.
    """

    alpha: float
    Do: float
    De: float
    intersections: int
    units: int


def segments(
    fragments: pd.DataFrame,
    *,
    coder: str = "codedBy",
    value: str = "code",
) -> dict[str, list[Segment]]:
    """Turn the position columns into per-coder segments.

    Only ``positionKind == "text"`` offers a continuum to measure boundaries
    on.  PDF rectangles do not, and are dropped with a warning rather than
    quietly approximated into something that looks like a character offset.

    The position columns arrived with a later plugin version; an export that
    predates them raises, rather than returning an empty result that would
    read as total disagreement.
    """
    needed = ["positionKind", "positionStart", "positionEnd"]
    missing = [c for c in needed if c not in fragments.columns]
    if missing:
        raise KeyError(
            f"this export predates the position columns (missing: "
            f"{', '.join(missing)}); re-export from a current zotQDA to "
            f"measure unitizing"
        )
    if (fragments["positionKind"] == "pdf").any():
        warnings.warn(
            "PDF segments have no linear continuum and were dropped; "
            "unitizing measures apply to text sources",
            stacklevel=2,
        )
    start = pd.to_numeric(fragments["positionStart"], errors="coerce")
    end = pd.to_numeric(fragments["positionEnd"], errors="coerce")
    keep = ((fragments["positionKind"] == "text")
            & start.notna() & end.notna() & (end > start))
    d = fragments[keep]
    out: dict[str, list[Segment]] = {}
    for who, group in d.groupby(d[coder].astype(str), sort=True):
        out[who] = sorted(
            ({"start": float(s), "end": float(e), "value": str(v)}
             for s, e, v in zip(start[group.index], end[group.index],
                                group[value].astype(str), strict=True)),
            key=lambda u: u["start"],
        )
    return out


def _overlap(a: Segment, b: Segment) -> float:
    """Length the two segments share; zero when they do not touch."""
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def _union(a: Segment, b: Segment) -> float:
    """Length from the earlier start to the later end.

    The span they jointly cover, gap included -- not the sum of their
    lengths, which is what the equations are written in terms of.
    """
    return max(a["end"], b["end"]) - min(a["start"], b["start"])


def _pair_disorder(
    a: Sequence[Segment],
    b: Sequence[Segment],
    delta2: Callable[[object, object], float],
) -> tuple[float, int]:
    """Equation (16) for ONE pair of coders.

    Every unit of ``a`` against every unit of ``b`` it touches.  A unit the
    other coder did not mark at all sits inside one of their gaps, and the
    equation charges twice its length -- in both directions, which is what
    the second loop covers.

    Gaps are never compared with each other: two coders agreeing that a
    stretch is irrelevant is not evidence of reliable unitizing.
    """
    total, n = 0.0, 0
    for u in a:
        matched = False
        for v in b:
            if _overlap(u, v) <= 0:
                continue
            matched = True
            total += _union(u, v) - _overlap(u, v) * (
                1 - delta2(u["value"], v["value"]))
            n += 1
        if not matched:
            total += 2 * (u["end"] - u["start"])
            n += 1
    for v in b:
        if not any(_overlap(u, v) > 0 for u in a):
            total += 2 * (v["end"] - v["start"])
            n += 1
    return total, n


def _observed_disorder(
    coders: Sequence[Sequence[Segment]],
    delta2: Callable[[object, object], float],
) -> tuple[float, int]:
    """Equation (17): the pairwise disorder summed over all coder pairs.

    Returns the summed disorder and the number of intersections behind it
    rather than their quotient, because the count is reported as well.
    """
    total, n = 0.0, 0
    for i in range(len(coders) - 1):
        for j in range(i + 1, len(coders)):
            pair_total, pair_n = _pair_disorder(coders[i], coders[j], delta2)
            total += pair_total
            n += pair_n
    return total, n


def _expected_disorder(
    units: Sequence[Segment],
    delta2: Callable[[object, object], float],
) -> float:
    """Equation (18): the disorder of units paired without regard to place.

    Every unit of every coder against every other, itself excluded, ignoring
    where on the continuum they sit.  This is the chance baseline alpha
    corrects against.  Returns ``nan`` when the denominator vanishes, which
    happens only for units of zero total length.
    """
    num = den = 0.0
    for a, ua in enumerate(units):
        la = ua["end"] - ua["start"]
        for b, ub in enumerate(units):
            if a == b:
                continue
            lb = ub["end"] - ub["start"]
            num += la * la + lb * lb + la * lb * delta2(ua["value"],
                                                        ub["value"])
            den += la + lb
    return num / den if den else math.nan


def unitizing_alpha(
    by_coder: Sequence[Sequence[Segment]],
    metric: Callable[[object, object], float] | None = None,
) -> AlphaResult | None:
    """Krippendorff's alpha for unitizing.

    Krippendorff (1995) doi:10.2307/271061, in the form given in the
    replacement of section 12.4 of *Content Analysis* (3rd ed.), equations 16
    to 19.  Gaps are not compared with each other: two coders agreeing that a
    stretch is irrelevant is not evidence of reliable unitizing.

    Parameters
    ----------
    by_coder:
        One list of segments per coder; a segment is
        ``{"start", "end", "value"}`` and one coder's segments must not
        overlap each other.
    metric:
        Squared difference between two values, nominal by default.  Pass
        ``lambda a, b: 0`` to measure identification alone.

        Ignoring the categories lowers the observed disagreement, but it
        lowers the *expected* disagreement too, because randomly paired
        units no longer differ by category either.  Which effect wins
        depends on whether the coders actually disagreed about categories:
        where they did, alpha rises; where they agreed throughout, alpha can
        fall.  Compare the two ``Do`` values, not the two alphas.

    Returns
    -------
    dict or None
        ``alpha``, the observed and expected disagreement ``Do`` and ``De``,
        the ``intersections`` behind ``Do``, and the number of ``units``.
        ``None`` when fewer than two coders contributed.
    """
    delta2 = metric if callable(metric) else (
        lambda c, k: 0.0 if str(c) == str(k) else 1.0)
    coders = [sorted((u for u in (units or [])
                      if u["end"] > u["start"]), key=lambda u: u["start"])
              for units in by_coder]
    coders = [c for c in coders if c]
    if len(coders) < 2:
        return None

    sum_o, n_o = _observed_disorder(coders, delta2)
    if not n_o:
        return None
    Do = sum_o / n_o

    every = [u for c in coders for u in c]
    if len(every) < 2:
        return None
    De = _expected_disorder(every, delta2)
    if not De or not math.isfinite(De):
        return None
    return {"alpha": 1 - Do / De, "Do": Do, "De": De,
            "intersections": n_o, "units": len(every)}


def _boundaries(units: Sequence[Segment], length: int) -> list[int]:
    """A segmentation as a 0/1 vector of boundary positions.

    Both ends of every segment mark a boundary. Position 0 and the very end
    are excluded: every segmentation shares those, so counting them would
    flatter the agreement by exactly as much on both sides.
    """
    v = [0] * max(0, length)
    for u in units or []:
        for pos in (u["start"], u["end"]):
            p = int(pos)
            if 0 < p < len(v):
                v[p] = 1
    return v


def _in_window(vec: list[int], i: int, k: int) -> int:
    """How many boundaries lie strictly inside the window at ``i``."""
    return sum(vec[i + 1:min(i + k + 1, len(vec))])


def _window_size(vec: list[int], length: int) -> int:
    """The conventional window: half the mean reference segment length."""
    # floor(x + 0.5), not round(): Python and R round halves to even while
    # JavaScript rounds them up, and a window width that differs by one
    # gives a different error rate. All three implementations must pick the
    # same width or their numbers are not comparable.
    return max(1, math.floor(length / (sum(vec) + 1) / 2 + 0.5))


def window_diff(
    reference: Sequence[Segment],
    hypothesis: Sequence[Segment],
    length: int,
    k: int | None = None,
) -> float:
    """WindowDiff: how often do the two disagree about the NUMBER of
    boundaries in a sliding window?

    Pevzner and Hearst (2002) doi:10.1162/089120102317341756.  Unlike
    :func:`pk` it charges for a spurious boundary as well as a missing one,
    and it treats a near miss as nearly right.  Zero means the boundaries
    coincide; it is an error rate, not an agreement.
    """
    return _sliding(reference, hypothesis, length, k, count=True)


def pk(
    reference: Sequence[Segment],
    hypothesis: Sequence[Segment],
    length: int,
    k: int | None = None,
) -> float:
    """Pk: the probability that two positions k apart are wrongly judged to
    lie in the same segment, or wrongly in different ones.

    Beeferman, Berger and Lafferty (1999) doi:10.1023/A:1007506220214.
    Reported next to :func:`window_diff` because the two disagree
    informatively: Pk is blind to how many boundaries a window holds, only to
    whether it holds one.
    """
    return _sliding(reference, hypothesis, length, k, count=False)


def _sliding(
    reference: Sequence[Segment],
    hypothesis: Sequence[Segment],
    length: int,
    k: int | None,
    *,
    count: bool,
) -> float:
    """The shared engine of :func:`window_diff` and :func:`pk`.

    Both slide the same window over the same two boundary vectors; they
    differ only in what counts as a disagreement inside it. ``count=True``
    compares the NUMBER of boundaries (WindowDiff), ``count=False`` only
    whether there is one at all (Pk). Keeping them in one function is what
    guarantees the two are measured over identical windows.

    Returns ``nan`` rather than a number when the text is too short or the
    window does not fit: an error rate computed over no windows is not zero.
    """
    L = int(length)
    if L < 2:
        return math.nan
    ref = _boundaries(reference, L)
    hyp = _boundaries(hypothesis, L)
    width = int(k) if k else _window_size(ref, L)
    if width < 1 or width >= L:
        return math.nan
    wrong = n = 0
    for i in range(0, L - width):
        a, b = _in_window(ref, i, width), _in_window(hyp, i, width)
        if (a != b) if count else ((a == 0) != (b == 0)):
            wrong += 1
        n += 1
    return wrong / n if n else math.nan
