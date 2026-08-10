"""Confidence intervals, per-code agreement and the kappa diagnostics.

Everything here exists because a single pooled coefficient with no interval
is easy to over-read.  Sim and Wright (2005) doi:10.1093/ptj/85.3.257 and
Zapf et al. (2016) doi:10.1186/s12874-016-0200-9 both make the point; this
module is the answer to it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypedDict

import pandas as pd

from . import agreement as ag

__all__ = [
    "agreement_by_code",
    "bootstrap_ci",
    "mulberry32",
    "paradox",
    "wilson",
]


class Interval(TypedDict):
    """A bootstrap interval, with the two numbers that make it checkable.

    ``used`` is how many resamples produced a finite value; it can be lower
    than ``resamples`` when a code is so rare that most resamples contain
    none of it, and that is exactly when the interval matters most.
    """

    estimate: float
    lo: float
    hi: float
    used: int
    resamples: int
    seed: int


class Proportion(TypedDict):
    """A proportion with a Wilson interval."""

    estimate: float
    lo: float
    hi: float
    n: float


class Paradox(TypedDict):
    """Why a kappa is disappointing (Feinstein and Cicchetti 1990)."""

    categories: list[str]
    n: int
    prevalence_index: float
    bias_index: float
    pabak: float
    percent: float
    table: dict[str, int]


def mulberry32(seed: int) -> Callable[[], float]:
    """The generator the plugin uses, reimplemented rather than borrowed.

    Python's own ``random`` would give different numbers, and then a
    bootstrap interval reported by qdaPy, qdaR and the plugin would be three
    different intervals for the same data with no way to tell which to
    believe.  This one is bit-for-bit identical across all three.
    """
    state = seed & 0xFFFFFFFF

    def rand() -> float:
        """The next value in [0, 1), advancing the 32-bit state."""
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t ^ (t + ((t ^ (t >> 7)) * (61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF)
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rand


def bootstrap_ci(
    matrix: pd.DataFrame,
    fn: Callable[[pd.DataFrame], float] = ag.fleiss,
    *,
    resamples: int = 1000,
    seed: int = 42,
    level: float = 0.95,
) -> Interval | None:
    """A percentile bootstrap interval for an agreement coefficient.

    Units are resampled, not ratings: the unit of analysis is the segment,
    and resampling ratings would treat two judgements of one segment as
    independent observations.

    Returns ``None`` rather than an interval when fewer than twenty
    resamples produced a finite value.  A wide interval is informative; one
    computed from almost nothing is not.
    """
    n = len(matrix)
    if not n:
        return None
    rand = mulberry32(seed)
    values: list[float] = []
    for _ in range(resamples):
        idx = [min(n - 1, int(rand() * n)) for _ in range(n)]
        sample = matrix.iloc[idx]
        sample.attrs["multi"] = matrix.attrs.get("multi", 0)
        try:
            v = fn(sample)
        except (ValueError, ZeroDivisionError):
            continue
        if isinstance(v, float) and math.isfinite(v):
            values.append(v)
    if len(values) < 20:
        return None
    values.sort()
    alpha = 1 - level

    def at(p: float) -> float:
        """The value at percentile ``p`` of the sorted resample values."""
        return values[min(len(values) - 1, max(0, int(p * len(values))))]

    return {
        "estimate": fn(matrix),
        "lo": at(alpha / 2),
        "hi": at(1 - alpha / 2),
        "used": len(values),
        "resamples": resamples,
        "seed": seed,
    }


def agreement_by_code(
    fragments: pd.DataFrame,
    *,
    min_n: int = 3,
    unit: str = "annotationKey",
    coder: str = "codedBy",
    value: str = "code",
) -> pd.DataFrame:
    """Agreement asked once per code, as a yes/no judgement.

    A pooled coefficient hides which codes the coders actually argued about.
    This is also the only honest treatment of material where segments
    legitimately carry several codes: each code is a separate question, so a
    segment with three codes contributes to three of them.

    ``min_n`` drops codes used fewer times than that; with two or three uses
    every coefficient is noise.
    """
    counts = fragments.loc[
        fragments[value].astype(str).str.len() > 0, value
    ].value_counts()
    codes = [c for c, n in counts.items() if n >= min_n]
    rows = []
    for code in codes:
        b = ag.units_binary(fragments, code, unit=unit, coder=coder, value=value)
        yes = int((b == "yes").to_numpy().sum())
        ci = wilson(yes, b.size)
        rows.append({
            "code": code,
            "n": int(counts[code]),
            "units": len(b),
            "percent": ag.percent_agreement(b),
            "cohen": ag.kappa(b) if b.shape[1] == 2 else math.nan,
            "ac1": ag.ac1(b),
            "prevalence": ci["estimate"],
            "lo": ci["lo"],
            "hi": ci["hi"],
        })
    out = pd.DataFrame(rows, columns=["code", "n", "units", "percent", "cohen",
                                      "ac1", "prevalence", "lo", "hi"])
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def paradox(matrix: pd.DataFrame) -> Paradox | None:
    """Why a kappa is disappointing.

    Feinstein and Cicchetti (1990) doi:10.1016/0895-4356(90)90158-L named the
    two reasons a kappa collapses while observed agreement is high: a skewed
    marginal distribution, and a systematic difference between the coders.
    The prevalence and bias indices measure exactly those, and PABAK is kappa
    recomputed with chance fixed at one half.

    Kappa alone tells a reader that agreement is poor.  These say why, which
    is the difference between a result and something to act on.

    Returns ``None`` for anything other than two coders and two categories,
    rather than a number that does not mean what it looks like.
    """
    if matrix.shape[1] != 2:
        return None
    pairs = [(str(x), str(y)) for x, y in matrix.itertuples(index=False)
             if not ag.is_missing(x) and not ag.is_missing(y)]
    if not pairs:
        return None
    cats = sorted({v for pair in pairs for v in pair})
    if len(cats) != 2:
        return None
    c1, c2 = cats
    n = len(pairs)
    a = sum(1 for x, y in pairs if x == c1 and y == c1)
    b = sum(1 for x, y in pairs if x == c1 and y == c2)
    c = sum(1 for x, y in pairs if x == c2 and y == c1)
    d = sum(1 for x, y in pairs if x == c2 and y == c2)
    po = (a + d) / n
    return {
        "categories": [c1, c2],
        "n": n,
        "prevalence_index": abs(a - d) / n,
        "bias_index": abs(b - c) / n,
        "pabak": 2 * po - 1,
        "percent": po,
        "table": {"a": a, "b": b, "c": c, "d": d},
    }


def wilson(successes: float, total: float, *,
           level: float = 0.95) -> Proportion:
    """A proportion with an interval that behaves at the edges.

    Wilson (1927) doi:10.1080/01621459.1927.10502953 rather than the textbook
    normal approximation, which Brown, Cai and DasGupta (2001)
    doi:10.1214/ss/1009213286 show to be erratic for small samples and
    degenerate at nought or one.  Code prevalences live exactly there: a code
    used in two of forty segments must not get an interval reaching below
    zero.
    """
    from scipy import stats as sps

    k, n = float(successes), float(total)
    if not math.isfinite(k) or not math.isfinite(n) or n <= 0 or k < 0 or k > n:
        return {"estimate": math.nan, "lo": math.nan, "hi": math.nan,
                "n": n if math.isfinite(n) else 0}
    z = float(sps.norm.ppf(1 - (1 - level) / 2))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # the Wilson interval always contains the point estimate; at p = 0 or 1
    # floating point can put a bound a machine epsilon on the wrong side, so
    # the containment is enforced rather than left to luck
    return {"estimate": p,
            "lo": min(p, max(0.0, centre - half)),
            "hi": max(p, min(1.0, centre + half)), "n": n}
