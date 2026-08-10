"""The statistics the plugins leave out on purpose.

qdaZ describes: frequencies, co-occurrence, agreement.  It does not test,
because a test invites a claim the design often does not support.  Where the
design *does* support one, this is where it is made -- explicitly, by someone
who chose the test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from .frames import code_matrix

__all__ = [
    "CAResult",
    "ChiSquareResult",
    "ClusterResult",
    "MDSResult",
    "ca",
    "ca_points",
    "chisq",
    "cluster",
    "code_distance",
    "mds",
    "mds_points",
]


@dataclass
class ChiSquareResult:
    """Outcome of :func:`chisq`."""

    table: pd.DataFrame
    statistic: float
    p_value: float
    dof: int
    cramers_v: float
    expected_ok: bool
    test: str
    n: int
    expected: pd.DataFrame = field(repr=False)

    def __str__(self) -> str:  # pragma: no cover - convenience only
        """The one-line summary, with the test named alongside its numbers.

        Which test ran matters as much as the p-value here: the exact and
        the Monte Carlo branch answer the same question differently.
        """
        return (f"{self.test}: stat={self.statistic:.4g}, p={self.p_value:.4g}, "
                f"Cramer's V={self.cramers_v:.3f}, "
                f"expected counts ok: {self.expected_ok}")


def _simulate_p(table: np.ndarray, statistic: float, *, resamples: int,
                seed: int) -> float:
    """Monte Carlo p-value for an r-by-c table with the margins fixed.

    The labels of one dimension are permuted, which keeps both sets of
    marginals and destroys only the association -- the same construction R's
    ``fisher.test(simulate.p.value = TRUE)`` uses.
    """
    rows = np.repeat(np.arange(table.shape[0]), table.sum(axis=1).astype(int))
    cols = np.repeat(np.arange(table.shape[1]), table.sum(axis=0).astype(int))
    rng = np.random.default_rng(seed)
    shape = table.shape
    at_least = 1  # the observed table counts as one, as it must
    for _ in range(resamples):
        shuffled = rng.permutation(cols)
        sim = np.zeros(shape, dtype=np.int64)
        np.add.at(sim, (rows, shuffled), 1)
        stat = sps.chi2_contingency(sim, correction=False)[0]
        if stat >= statistic - 1e-12:
            at_least += 1
    return at_least / (resamples + 1)


def chisq(
    fragments: pd.DataFrame,
    group: str | pd.Series = "citekey",
    *,
    codes: list[str] | None = None,
    resamples: int = 2000,
    seed: int = 42,
) -> ChiSquareResult:
    """Test whether codes are distributed independently of a grouping.

    Reports Cramer's V as an effect size, and says whether the chi-squared
    approximation was appropriate at all: when expected counts fall below
    five, the exact test is reported instead -- Fisher's for a two-by-two
    table, a Monte Carlo p-value with the margins fixed for anything larger.
    ``expected_ok`` is part of the result rather than a warning to be missed.

    Note what the unit of this test is: one coded fragment.  Fragments from
    the same document are not independent observations, so a significant
    result across documents is weaker evidence than the p-value suggests.
    """
    values = (fragments[group] if isinstance(group, str) else pd.Series(group))
    if len(values) != len(fragments):
        raise ValueError("group must have one value per fragment")
    keep = fragments["code"].astype(str).str.len() > 0
    if codes is not None:
        keep &= fragments["code"].isin(codes)
    table = pd.crosstab(fragments.loc[keep, "code"].astype(str),
                        values[keep].astype(str))
    if min(table.shape) < 2:
        raise ValueError(
            "need at least two codes and two groups to test independence"
        )

    counts = table.to_numpy(dtype=np.int64)
    stat, p, dof, expected = sps.chi2_contingency(counts)
    expected_ok = bool((expected >= 5).all())
    n = int(counts.sum())
    v = math.sqrt(stat / (n * (min(counts.shape) - 1)))

    test = "chi-squared test of independence"
    if not expected_ok:
        raw_stat = sps.chi2_contingency(counts, correction=False)[0]
        if counts.shape == (2, 2):
            p = float(sps.fisher_exact(counts)[1])
            test = "Fisher's exact test (expected counts below 5)"
        else:
            p = _simulate_p(counts, raw_stat, resamples=resamples, seed=seed)
            test = (f"Monte Carlo test with fixed margins, {resamples} "
                    f"resamples, seed {seed} (expected counts below 5)")

    return ChiSquareResult(
        table=table, statistic=float(stat), p_value=float(p), dof=int(dof),
        cramers_v=float(v), expected_ok=expected_ok, test=test, n=n,
        expected=pd.DataFrame(expected, index=table.index,
                              columns=table.columns),
    )


def code_distance(
    fragments: pd.DataFrame,
    *,
    unit: str = "annotationKey",
    min_n: int = 3,
) -> pd.DataFrame:
    """Jaccard distances between codes.

    Two codes are close when they are assigned to the same segments.  This is
    what the scaling and the clustering below work on, and the same
    coefficient the plugin uses to propose code matches.

    Parameters
    ----------
    min_n:
        Ignore codes used fewer than this many times.  Rarely used codes make
        every coefficient unstable.
    """
    if unit not in fragments.columns:
        raise KeyError(f"fragments is missing the column {unit!r}")
    coded = fragments[fragments["code"].astype(str).str.len() > 0]
    incidence = pd.crosstab(coded[unit].astype(str),
                            coded["code"].astype(str)) > 0
    incidence = incidence.loc[:, incidence.sum(axis=0) >= min_n]
    if incidence.shape[1] < 2:
        raise ValueError(f"fewer than two codes reach min_n = {min_n}")

    m = incidence.to_numpy(dtype=bool)
    inter = m.T.astype(int) @ m.astype(int)
    sizes = m.sum(axis=0)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(union == 0, 1.0, 1.0 - inter / np.where(union == 0, 1, union))
    np.fill_diagonal(dist, 0.0)
    names = list(incidence.columns)
    return pd.DataFrame(dist, index=names, columns=names)


@dataclass
class MDSResult:
    """Outcome of :func:`mds`."""

    points: pd.DataFrame
    goodness: tuple[float, float]


def mds(
    fragments: pd.DataFrame,
    *,
    unit: str = "annotationKey",
    min_n: int = 3,
    k: int = 2,
) -> MDSResult:
    """Classical (Torgerson) scaling of the code distances.

    Places codes in two dimensions so that codes applied to the same segments
    end up close together.  A map of this kind says nothing about
    significance; it is a way of looking at a distance matrix.  ``goodness``
    is the share of the eigenvalue mass the dimensions carry -- read it before
    reading the map.
    """
    d = code_distance(fragments, unit=unit, min_n=min_n)
    names = list(d.index)
    n = len(names)
    k = min(k, n - 1)
    D2 = d.to_numpy(dtype=float) ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * centering @ D2 @ centering
    values, vectors = np.linalg.eigh(B)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    used = values[:k]
    coords = vectors[:, :k] * np.sqrt(np.clip(used, 0, None))
    points = pd.DataFrame(
        coords, columns=[f"dim{i + 1}" for i in range(k)]
    )
    points.insert(0, "code", names)
    total_abs = np.abs(values).sum()
    total_pos = np.clip(values, 0, None).sum()
    goodness = (
        float(np.abs(used).sum() / total_abs) if total_abs else math.nan,
        float(np.clip(used, 0, None).sum() / total_pos) if total_pos else math.nan,
    )
    return MDSResult(points=points, goodness=goodness)


@dataclass
class ClusterResult:
    """Outcome of :func:`cluster`."""

    linkage: np.ndarray = field(repr=False)
    labels: list[str] = field(repr=False)
    distance: pd.DataFrame = field(repr=False)
    cophenetic: float

    def flat(self, n_clusters: int) -> pd.DataFrame:
        """Cut the tree into ``n_clusters`` groups."""
        assignment = hierarchy.fcluster(self.linkage, n_clusters,
                                        criterion="maxclust")
        return pd.DataFrame({"code": self.labels, "cluster": assignment})


def cluster(
    fragments: pd.DataFrame,
    *,
    unit: str = "annotationKey",
    min_n: int = 3,
    method: str = "average",
) -> ClusterResult:
    """Cluster codes by the segments they share.

    The cophenetic correlation comes with it, because a dendrogram always
    looks convincing even when it represents the distances poorly; values well
    below about 0.7 mean the picture should not be over-read.  With fewer than
    three codes the correlation is undefined and reported as ``nan`` rather
    than as a number that means nothing.
    """
    d = code_distance(fragments, unit=unit, min_n=min_n)
    condensed = squareform(d.to_numpy(dtype=float), checks=False)
    link = hierarchy.linkage(condensed, method=method)
    if len(d.index) < 3:
        coph = math.nan
    else:
        coph = float(hierarchy.cophenet(link, condensed)[0])
    return ClusterResult(linkage=link, labels=list(d.index), distance=d,
                         cophenetic=coph)


@dataclass
class CAResult:
    """Outcome of :func:`ca`."""

    row_scores: pd.DataFrame
    col_scores: pd.DataFrame
    row_coords: pd.DataFrame
    col_coords: pd.DataFrame
    inertia: np.ndarray
    inertia_share: np.ndarray
    total_inertia: float
    table: pd.DataFrame = field(repr=False)


def ca(
    fragments: pd.DataFrame,
    *,
    doc_col: str = "citekey",
    n_dims: int = 2,
) -> CAResult:
    """Correspondence analysis of the code-by-document table.

    Shows which codes and which documents attract each other.  Unlike the
    plugin's descriptive matrix this decomposes the table and reports how much
    of its inertia the dimensions explain -- the honest answer to "how much of
    the picture am I actually seeing".

    ``row_scores`` and ``col_scores`` are standard coordinates (comparable to
    ``MASS::corresp`` in R); ``row_coords`` and ``col_coords`` are principal
    coordinates, which is what you plot.
    """
    table = code_matrix(fragments, doc_col=doc_col, long=False)
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if min(table.shape) < 2:
        raise ValueError("correspondence analysis needs at least a 2x2 table")

    N = table.to_numpy(dtype=float)
    P = N / N.sum()
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)

    keep = min(n_dims, min(table.shape) - 1)
    sv_k = sv[:keep]
    names = [f"dim{i + 1}" for i in range(keep)]
    row_std = U[:, :keep] / np.sqrt(r)[:, None]
    col_std = Vt[:keep, :].T / np.sqrt(c)[:, None]

    return CAResult(
        row_scores=pd.DataFrame(row_std, index=table.index, columns=names),
        col_scores=pd.DataFrame(col_std, index=table.columns, columns=names),
        row_coords=pd.DataFrame(row_std * sv_k, index=table.index,
                                columns=names),
        col_coords=pd.DataFrame(col_std * sv_k, index=table.columns,
                                columns=names),
        inertia=sv_k ** 2,
        inertia_share=sv_k ** 2 / (sv ** 2).sum(),
        total_inertia=float((sv ** 2).sum()),
        table=table,
    )


# ---------------------------------------------------------------------------
# Plotting-ready views
#
# Both drawing backends need the same reshaping before they can plot a map,
# and both used to carry their own copy of it. That is the one place where
# `qdapy.frames`' promise -- the backends cannot differ in WHAT they show,
# only in how -- was not kept. These live here rather than in `frames`
# because they build on `ca()` and `mds()`, and `frames` is what `stats`
# itself is built on: putting them there would make the two modules import
# each other.


def ca_points(fragments: pd.DataFrame, *,
              doc_col: str = "citekey") -> pd.DataFrame:
    """Codes and documents as one table of plottable points.

    The principal coordinates of both, stacked, with ``kind`` saying which
    is which and ``label`` carrying the name. ``df.attrs["inertia_shown"]``
    is the share of the table's inertia the two dimensions carry -- a map
    that shows a fifth of the structure looks exactly like one that shows
    all of it, so a caption without that number is misleading.
    """
    result = ca(fragments, doc_col=doc_col)
    rows = result.row_coords.reset_index().rename(columns={"code": "label"})
    rows["kind"] = "code"
    cols = result.col_coords.reset_index().rename(
        columns={"document": "label"})
    cols["kind"] = "document"
    out = pd.concat([rows, cols], ignore_index=True)
    out.attrs["inertia_shown"] = float(result.inertia_share[:2].sum())
    return out


def mds_points(fragments: pd.DataFrame, *, unit: str = "annotationKey",
               min_n: int = 3) -> pd.DataFrame:
    """Codes as plottable points from the classical scaling.

    A second dimension is added as a column of zeros when the scaling could
    only produce one, so a caller never has to guess whether ``dim2`` is
    there. ``df.attrs["goodness"]`` is the first of the two goodness
    figures; read it before reading the map.
    """
    result = mds(fragments, unit=unit, min_n=min_n)
    out = result.points.copy()
    if "dim2" not in out.columns:
        out["dim2"] = 0.0
    out.attrs["goodness"] = float(result.goodness[0])
    return out
