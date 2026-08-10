"""Planning a study, rather than only analysing one.

Every other module here answers a question about material you already have.
These two answer the question you face before collecting it, which no
established QDA tool asks at all: how much material will it take?
"""

from __future__ import annotations

import math

__all__ = [
    "kappa_lower",
    "plan_kappa",
    "plan_themes",
    "theme_power",
]


def _cell_probs(kappa: float, prevalence: float,
                raters: int) -> tuple[float, float, float]:
    """Donner and Rotondi's common-correlation model, collapsed to three cells.

    Their reparameterisation (equations 2-3, following Altaye et al.) leaves
    only: every rater says no, every rater says yes, anything in between.
    """
    p, n = prevalence, raters
    none = (1 - p) ** n * (1 - kappa) + kappa * (1 - p)
    every = p ** n * (1 - kappa) + kappa * p
    return none, 1 - none - every, every


def _chisq_per_subject(kappa0: float, kappa_lower_: float, prevalence: float,
                       raters: int) -> float:
    """The Pearson statistic of their equation (4), per subject.

    All disagreeing cells are pooled, which leaves one degree of freedom and
    makes the statistic linear in the sample size -- so the answer comes out
    in closed form instead of by iteration.
    """
    observed = _cell_probs(kappa0, prevalence, raters)
    under_null = _cell_probs(kappa_lower_, prevalence, raters)
    return sum((o - e) ** 2 / e for o, e in zip(observed, under_null, strict=True)
               if e > 0)


def plan_kappa(kappa0: float, kappa_lower_target: float, prevalence: float,
               *, raters: int = 2, alpha: float = 0.05,
               critical: float | None = None) -> float:
    """How many segments must be double-coded?

    Donner and Rotondi (2010) doi:10.2202/1557-4679.1275 give the sample size
    at which the *lower* bound of a one-sided interval for kappa reaches a
    value named in advance. That is the quantity a reader cares about, since
    nobody ever objected that agreement was too good.

    Prevalence matters more than people expect: a code applied to a tenth of
    the segments needs several times the material of one applied to a third.
    The requirement is symmetric about 0.5, so a conservative planner takes
    the value further from it.

    ``critical`` defaults to the exact chi-squared quantile. The published
    tables used it rounded to 2.71, which makes eight of their forty-eight
    cells one larger; pass ``critical=2.71`` to reproduce them exactly.

    Returns ``inf`` when the target is not below the anticipated kappa: no
    sample size makes an interval reach a bound at or above its own centre.
    """
    from scipy import stats as sps

    if not (0 < prevalence < 1) or raters < 2:
        raise ValueError("prevalence must be in (0, 1) and raters at least 2")
    if kappa_lower_target >= kappa0:
        return math.inf
    crit = critical if critical is not None else float(
        sps.chi2.ppf(1 - 2 * alpha, df=1))
    per = _chisq_per_subject(kappa0, kappa_lower_target, prevalence, raters)
    if not math.isfinite(per) or per <= 0:
        return math.inf
    return math.ceil(crit / per)


def kappa_lower(n: int, kappa0: float, prevalence: float, *, raters: int = 2,
                alpha: float = 0.05, critical: float | None = None) -> float:
    """What lower bound can this much material reach?

    The same calculation the other way round, which is the question you face
    when the number of segments is already fixed by the budget.
    """
    from scipy import optimize
    from scipy import stats as sps

    if n <= 0 or not (0 < prevalence < 1) or raters < 2:
        raise ValueError("n must be positive, prevalence in (0, 1), raters >= 2")
    crit = critical if critical is not None else float(
        sps.chi2.ppf(1 - 2 * alpha, df=1))

    def target(kl: float) -> float:
        """Zero where the lower confidence bound sits; bisected below."""
        return n * _chisq_per_subject(kappa0, kl, prevalence, raters) - crit

    lo = -0.999
    if target(lo) < 0:
        return math.nan          # not even a kappa of nought can be excluded
    hi = kappa0 - 1e-9
    if target(hi) > 0:
        return kappa0
    return float(optimize.brentq(target, lo, hi, xtol=1e-12))


def plan_themes(prevalence: float, *, instances: int = 1, power: float = 0.8,
                max_n: int = 10000) -> int | None:
    """How many documents to be reasonably sure of meeting a theme?

    Fugard and Potts (2015) doi:10.1080/13645579.2015.1005453 treat the
    waiting time as negative binomial, which is the same as requiring the
    binomial tail ``P(X >= instances)`` to reach the desired power.

    **What it assumes, and who disputes it.** Themes are present or absent,
    independent of one another, and certain to surface once present. Braun and
    Clarke (2016) doi:10.1080/13645579.2016.1195588 reject the premise for
    reflexive thematic analysis, where themes are constructed rather than
    discovered and a population prevalence is not a meaningful quantity. This
    is a planning aid for work that accepts those assumptions, not a sample
    size requirement for qualitative research at large.
    """
    from scipy import stats as sps

    if not (0 < prevalence <= 1) or instances < 1 or not (0 < power < 1):
        raise ValueError("prevalence in (0, 1], instances >= 1, power in (0, 1)")
    for n in range(instances, max_n + 1):
        if float(sps.binom.sf(instances - 1, n, prevalence)) >= power:
            return n
    return None


def theme_power(n: int, prevalence: float, *, instances: int = 1) -> float:
    """The same question as power: how likely am I to meet it with what I have?"""
    from scipy import stats as sps

    if n < 0 or not (0 <= prevalence <= 1) or instances < 1:
        raise ValueError("n >= 0, prevalence in [0, 1], instances >= 1")
    return float(sps.binom.sf(instances - 1, n, prevalence))
