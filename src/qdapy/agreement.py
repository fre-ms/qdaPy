"""Intercoder reliability, recomputed independently of the plugin.

Reliability is where a second implementation earns its keep: the coefficients
here are computed from the exported file alone, in a language that shares
nothing with the plugin but the exchange contract.  A figure that ends up in a
methods section has therefore been produced twice.  The test suite checks this
package against frozen plugin results on randomly generated coder matrices.

Nominal data only, which is what codes are.

References
----------
Cohen (1960) doi:10.1177/001316446002000104
Fleiss (1971) doi:10.1037/h0031619
Brennan and Prediger (1981) doi:10.1177/001316448104100307
Gwet (2008) doi:10.1348/000711006X126600
Hayes and Krippendorff (2007) doi:10.1080/19312450709336664
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ac1",
    "agreement",
    "alpha",
    "brennan",
    "confusion",
    "flatten_path",
    "fleiss",
    "is_missing",
    "kappa",
    "level_agreement",
    "percent_agreement",
    "units",
    "units_binary",
]

NO_CODE = "(no code)"


def is_missing(value: object) -> bool:
    """Is this rating absent rather than a category?

    A coder who never saw a unit is missing data; a coder who assigned
    nothing is not the same thing, and Krippendorff's alpha is in this
    package partly because it tells the two apart.  ``None``, ``NaN`` and
    ``pandas.NA`` all mean absent, because all three reach us from a CSV.

    Public because :mod:`qdapy.reliability` needs the same notion and
    reaching across a module boundary for a private helper is exactly the
    coupling this package avoids elsewhere.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return value is pd.NA


def _rows(matrix: pd.DataFrame) -> list[list[str]]:
    """The ratings per unit, missing ones dropped."""
    out = []
    for row in matrix.itertuples(index=False):
        out.append([str(v) for v in row if not is_missing(v)])
    return out


def _categories(matrix: pd.DataFrame) -> list[str]:
    """Every category anyone actually used, sorted.

    Derived from the ratings rather than declared: a category nobody assigned
    is not part of the chance correction, because no coder could have picked
    it.
    """
    seen: set[str] = set()
    for vals in _rows(matrix):
        seen.update(vals)
    return sorted(seen)


def _multi(matrix: pd.DataFrame) -> int:
    """How many units :func:`units` had to set aside as multi-coded.

    Carried on ``attrs`` by the matrix builder and reported alongside the
    coefficients, because a coefficient computed on two thirds of the
    material should say so.
    """
    return int(matrix.attrs.get("multi", 0))


def flatten_path(path: str | Iterable[str],
                 level: int | None = None) -> str | list[str]:
    """Shorten a code path to a number of levels.

    ``"Belastung/beruflich/akut"`` at level 2 becomes
    ``"Belastung/beruflich"``.  ``None`` or ``0`` keeps everything.
    """
    def one(p: object) -> str:
        """Shorten a single path; a missing value flattens to ""."""
        text = "" if is_missing(p) else str(p)
        if not level or level < 1:
            return text
        return "/".join(text.split("/")[:level])

    if isinstance(path, str):
        return one(path)
    return [one(p) for p in path]


def units(  # noqa: C901 -- one branch per documented reshaping decision
    fragments: pd.DataFrame,
    uncoded: pd.DataFrame | None = None,
    *,
    unit: str = "annotationKey",
    coder: str = "codedBy",
    value: str = "code",
    no_code: str = NO_CODE,
    level: int | None = None,
) -> pd.DataFrame:
    """Build the unit-by-coder matrix the measures work on.

    The fragments export is longer than that -- one row per annotation and
    code -- so it has to be reshaped, and two decisions have to be made
    explicitly rather than by accident.

    **Segments nobody coded are a category, not a gap.**  Agreement about what
    is *not* relevant is agreement.  Pass the ``uncoded`` export and those
    segments enter as their own category; leave it out and the figures only
    describe the segments at least one coder marked, which is a different and
    usually more flattering question.

    **A segment one coder coded twice is set aside.**  Where a coder gave one
    segment several codes there is no single value to compare, so the cell
    becomes missing and is counted in ``matrix.attrs["multi"]``.  The honest
    way to include such material is the per-code binary view,
    :func:`units_binary`; reporting an overall figure that quietly dropped a
    tenth of the segments is not.

    Parameters
    ----------
    value:
        ``"code"`` is the readable path, ``"codeId"`` the identity that
        survives renaming, moving and merging.
    level:
        Flatten paths to this many levels first; see :func:`level_agreement`.

    Returns
    -------
    pandas.DataFrame
        Units in rows, coders in columns, ``None`` where a coder did not rate
        a unit, with ``attrs["multi"]`` counting the cells set aside.  A
        matrix you build yourself may use ``NaN`` or ``pd.NA`` instead; all
        three are understood as "not rated".
    """
    for col in (unit, coder, value):
        if col not in fragments.columns:
            raise KeyError(f"fragments is missing the column {col!r}")

    coded = fragments[fragments[value].astype(str).str.len() > 0]
    values = coded[value].astype(str)
    if level:
        values = pd.Series(flatten_path(values.tolist(), level), index=coded.index)

    coders = sorted(set(coded[coder].astype(str)))
    keys = list(dict.fromkeys(coded[unit].astype(str)))
    if uncoded is not None:
        for key in uncoded[unit].astype(str):
            if key not in keys:
                keys.append(key)

    if not keys or not coders:
        empty = pd.DataFrame(index=pd.Index(keys, name=unit), columns=coders,
                             dtype=object)
        empty.attrs["multi"] = 0
        return empty

    # every coder who coded anywhere is taken to have seen every segment, so
    # "coded nothing here" is a judgement and not a missing observation
    fill = no_code if uncoded is not None else None
    # np.full keeps None as None; passing None to the DataFrame constructor
    # would turn it into NaN and leave two different sentinels in one matrix
    matrix = pd.DataFrame(
        np.full((len(keys), len(coders)), fill, dtype=object),
        index=pd.Index(keys, name=unit), columns=coders,
    )

    cells: dict[tuple[str, str], set[str]] = defaultdict(set)
    for key, who, val in zip(coded[unit].astype(str), coded[coder].astype(str),
                             values, strict=True):
        cells[(key, who)].add(val)

    multi = 0
    for (key, who), vals in cells.items():
        if len(vals) > 1:
            multi += 1
            matrix.at[key, who] = None
        else:
            matrix.at[key, who] = next(iter(vals))

    matrix.attrs["multi"] = multi
    return matrix


def units_binary(
    fragments: pd.DataFrame,
    code: str,
    uncoded: pd.DataFrame | None = None,
    *,
    unit: str = "annotationKey",
    coder: str = "codedBy",
    value: str = "code",
) -> pd.DataFrame:
    """Turn one code into a yes/no judgement per unit.

    This is how multiply coded material can still be assessed: every code is
    asked about separately, so a segment carrying three codes contributes to
    all three questions instead of being dropped.
    """
    coders = sorted(
        set(fragments.loc[fragments[value].astype(str).str.len() > 0, coder]
            .astype(str))
    )
    keys = list(dict.fromkeys(fragments[unit].astype(str)))
    if uncoded is not None:
        for key in uncoded[unit].astype(str):
            if key not in keys:
                keys.append(key)

    matrix = pd.DataFrame("no", index=pd.Index(keys, name=unit),
                          columns=coders, dtype=object)
    hit = fragments[fragments[value].astype(str) == str(code)]
    for key, who in zip(hit[unit].astype(str), hit[coder].astype(str),
                        strict=True):
        if key in matrix.index and who in matrix.columns:
            matrix.at[key, who] = "yes"
    matrix.attrs["multi"] = 0
    return matrix


# --- the measures -----------------------------------------------------


def percent_agreement(matrix: pd.DataFrame) -> float:
    """The share of agreeing coder pairs, over all units both coders rated.

    Easy to read and, on its own, easy to over-read: with one dominant
    category a high value says very little.
    """
    agree = total = 0
    for vals in _rows(matrix):
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                total += 1
                if vals[i] == vals[j]:
                    agree += 1
    return agree / total if total else math.nan


def kappa(matrix: pd.DataFrame) -> float:
    """Cohen's kappa, for exactly two coders, on the units both rated.

    Chance is estimated from the coders' own marginals, which is what makes
    kappa fall when one category dominates -- the paradox that keeps being
    mistaken for a defect of the coding.  Returns ``nan`` where kappa is
    undefined rather than a number that means nothing.
    """
    if matrix.shape[1] != 2:
        raise ValueError(
            f"Cohen's kappa is for two coders; this matrix has "
            f"{matrix.shape[1]} -- use fleiss() or alpha()"
        )
    pairs = [v for v in (list(r) for r in matrix.itertuples(index=False))
             if not is_missing(v[0]) and not is_missing(v[1])]
    n = len(pairs)
    if not n:
        return math.nan
    a = [str(p[0]) for p in pairs]
    b = [str(p[1]) for p in pairs]
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in set(a) | set(b))
    if pe >= 1:
        return math.nan
    return (po - pe) / (1 - pe)


def brennan(matrix: pd.DataFrame, q: int | None = None) -> float:
    """Brennan and Prediger's kappa, with uniform chance ``1/q``.

    This is the figure MAXQDA reports, so it is the one to use when a result
    has to line up with a MAXQDA output.

    Parameters
    ----------
    q:
        Number of categories the scheme offers.  Defaults to the categories
        present anywhere in ``matrix`` -- pass the size of the code system
        when coders could have chosen codes they never used, because that is
        the number the coefficient is actually about.
    """
    if matrix.shape[1] != 2:
        raise ValueError(
            "Brennan and Prediger's kappa is defined here for two coders"
        )
    pairs = [v for v in (list(r) for r in matrix.itertuples(index=False))
             if not is_missing(v[0]) and not is_missing(v[1])]
    n = len(pairs)
    if not n:
        return math.nan
    k = max(2, q if q is not None else len(_categories(matrix)))
    po = sum(str(x) == str(y) for x, y in pairs) / n
    return (po - 1 / k) / (1 - 1 / k)


def fleiss(matrix: pd.DataFrame) -> float:
    """Fleiss' kappa, for any number of coders.

    Units rated by fewer than two coders carry no agreement information and
    are skipped.  Because it works on one category per unit, this is the
    coefficient that makes the case for coding a segment once: where segments
    routinely carry several codes there is no single value to compare, and the
    overall figure is then computed on whatever remains unambiguous.
    :func:`units` counts what it set aside, and that count belongs next to the
    kappa.
    """
    cats = _categories(matrix)
    if len(cats) < 2:
        return math.nan
    pa_sum = 0.0
    n_units = 0
    cat_total: Counter[str] = Counter()
    rating_total = 0
    for vals in _rows(matrix):
        r = len(vals)
        if r < 2:
            continue
        n_units += 1
        rating_total += r
        counts = Counter(vals)
        pa_sum += sum(c * (c - 1) for c in counts.values()) / (r * (r - 1))
        cat_total.update(counts)
    if not n_units or not rating_total:
        return math.nan
    pa = pa_sum / n_units
    pe = sum((cat_total[c] / rating_total) ** 2 for c in cats)
    if pe >= 1:
        return math.nan
    return (pa - pe) / (1 - pe)


def alpha(matrix: pd.DataFrame) -> float:
    """Krippendorff's alpha, nominal, via the coincidence matrix.

    Takes any number of coders and tolerates missing values, so a coder who
    skipped a unit costs that unit rather than the analysis.
    """
    o: defaultdict[tuple[str, str], float] = defaultdict(float)
    nc: defaultdict[str, float] = defaultdict(float)
    n = 0.0
    for vals in _rows(matrix):
        m = len(vals)
        if m < 2:
            continue
        w = 1 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                o[(vals[i], vals[j])] += w
                nc[vals[i]] += w
                n += w
    if n <= 1:
        return math.nan
    d_obs = sum(mass for (a, b), mass in o.items() if a != b)
    cats = list(nc)
    d_exp = sum(nc[a] * nc[b] for a in cats for b in cats if a != b) / (n - 1)
    if d_exp == 0:
        return math.nan
    return 1 - d_obs / d_exp


def ac1(matrix: pd.DataFrame) -> float:
    """Gwet's AC1, stable where one category dominates.

    Worth reporting beside kappa rather than instead of it: where the two
    diverge, the marginals are the story.
    """
    cats = _categories(matrix)
    q = len(cats)
    if q < 2:
        return math.nan
    pa_sum = 0.0
    n_units = 0
    pi_sum: dict[str, float] = dict.fromkeys(cats, 0.0)
    for vals in _rows(matrix):
        r = len(vals)
        if r < 2:
            continue
        n_units += 1
        counts = Counter(vals)
        pa_sum += sum(c * (c - 1) for c in counts.values()) / (r * (r - 1))
        for c in cats:
            pi_sum[c] += counts[c] / r
    if not n_units:
        return math.nan
    pa = pa_sum / n_units
    pe = sum((pi_sum[c] / n_units) * (1 - pi_sum[c] / n_units)
             for c in cats) / (q - 1)
    if pe >= 1:
        return math.nan
    return (pa - pe) / (1 - pe)


def agreement(matrix: pd.DataFrame) -> pd.DataFrame:
    """Report every measure side by side.

    No single coefficient settles the question: they disagree exactly where
    the marginals are skewed, and seeing them disagree is the finding.
    Cohen's and Brennan's are ``nan`` for more than two coders rather than
    silently computed on the first two columns.
    """
    two = matrix.shape[1] == 2
    comparable = sum(1 for vals in _rows(matrix) if len(vals) >= 2)
    return pd.DataFrame(
        [{
            "units": comparable,
            "coders": matrix.shape[1],
            "categories": len(_categories(matrix)),
            "multi_set_aside": _multi(matrix),
            "percent": percent_agreement(matrix),
            "cohen": kappa(matrix) if two else math.nan,
            "brennan": brennan(matrix) if two else math.nan,
            "fleiss": fleiss(matrix),
            "alpha": alpha(matrix),
            "ac1": ac1(matrix),
        }]
    )


def level_agreement(
    matrix: pd.DataFrame, max_level: int | None = None
) -> pd.DataFrame:
    """Agreement at each level of the code system.

    A hierarchical code system can be read at several resolutions, and coders
    who split over ``Belastung/beruflich`` against ``Belastung/privat`` still
    agree that the segment is about ``Belastung``.  Flattening the paths level
    by level and recomputing shows where in the hierarchy the agreement is
    lost -- which is a statement about the code system, not about the coders.
    """
    depth = max_level
    if depth is None:
        depths = [len(v.split("/")) for vals in _rows(matrix) for v in vals]
        depth = max(depths) if depths else 1
    frames = []
    for level in range(1, int(depth) + 1):
        flat = matrix.map(
            lambda v, lvl=level: None if is_missing(v) else flatten_path(str(v), lvl)
        )
        flat.attrs["multi"] = _multi(matrix)
        row = agreement(flat)
        row.insert(0, "level", level)
        frames.append(row)
    return pd.concat(frames, ignore_index=True)


def confusion(
    matrix: pd.DataFrame, *, only_disagreements: bool = False
) -> pd.DataFrame:
    """Where two coders disagreed, most frequent pair first.

    The confusion table is what turns a disappointing kappa into something
    actionable: usually a handful of category pairs account for most of it,
    and those pairs are the ones whose definitions need work.
    """
    if matrix.shape[1] != 2:
        raise ValueError("the confusion table is for two coders")
    left, right = list(matrix.columns)
    counts: Counter[tuple[str, str]] = Counter()
    for row in matrix.itertuples(index=False):
        a, b = list(row)
        if is_missing(a) or is_missing(b):
            continue
        counts[(str(a), str(b))] += 1
    rows: Sequence[tuple[tuple[str, str], int]] = counts.most_common()
    out = pd.DataFrame(
        [{left: a, right: b, "n": n} for (a, b), n in rows]
    )
    if only_disagreements and not out.empty:
        out = out[out[left] != out[right]].reset_index(drop=True)
    return out
