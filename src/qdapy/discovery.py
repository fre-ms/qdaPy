"""How saturated is this material, and how much more would it take?

A saturation curve that is still climbing says nothing about how far from the
top it is. Lowe, Norris, Farris and Babbage (2018)
doi:10.1177/1525822X17749386 fit the accumulation of themes to a growth model,
which estimates the number of themes there are to be found and thereby turns
"still climbing" into a percentage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["MODELS", "documents_for", "saturation_index"]

MODELS = ("IS", "IW", "SW")


def _curve(model: str, n: npt.ArrayLike, A: float, b: float) -> Any:
    """Lowe et al.'s three accumulation models.

    ``IS`` assumes observations are independent (a6), ``IW`` that overlap
    grows with what is already known (a10), ``SW`` that it grows with the
    number of observations (a12).
    """
    from scipy.special import gammaln

    n = np.asarray(n, dtype=float)
    if model == "IS":
        return A * (1 - (1 - b) ** n)
    if model == "IW":
        return A * b * n / (1 + b * (n - 1))
    if model == "SW":
        # the gamma ratio overflows past n of about 170, so it is taken in
        # logs; the quantity itself stays small, only the pieces are huge
        return A * (1 - np.exp(gammaln(1 - b + n) - gammaln(1 - b) - gammaln(1 + n)))
    raise ValueError(f"unknown model: {model}")


class Fit(TypedDict):
    """What :func:`saturation_index` reports.

    ``index`` is the share of the estimable themes already found.  When the
    curve could not be fitted it is NaN and ``reason`` says why; every other
    field is then meaningless and deliberately left at its empty value.
    """

    model: str
    A: float
    b: float
    index: float
    fitted: list[float]
    rmse: float
    reason: str


def saturation_index(cumulative: Sequence[float],
                     model: str = "IW") -> Fit:
    """Fit the accumulation curve and report how much of it you have.

    The index is the share of the estimable themes already found,
    ``100 * T_N / floor(A)``. Because it comes from a fitted ``A``, it also
    answers what a project halfway through actually asks: how many more
    documents for another ten points -- see :func:`documents_for`.

    Lowe et al. found no single best model, so fit all three and look at
    which describes your data rather than choosing one in advance. That they
    disagree is information, not a defect.
    """
    from scipy import optimize

    if isinstance(cumulative, pd.DataFrame):
        cumulative = cumulative["cumulative"]
    y = np.asarray(list(cumulative), dtype=float)
    n = np.arange(1, len(y) + 1, dtype=float)
    if len(y) < 3 or y.max() <= 0:
        return {"model": model, "A": math.nan, "b": math.nan,
                "index": math.nan, "fitted": [math.nan] * len(y),
                "rmse": math.nan,
                "reason": "fewer than three documents, or no themes"}

    # Lowe et al.'s direct IW estimate (a16) is exact algebra from the first
    # and last point, and makes a good starting value for the other models
    N, T1, TN = len(y), y[0], y[-1]
    denom = N * T1 - TN
    A0 = (N - 1) * T1 * TN / denom if denom != 0 else TN * 2
    if not math.isfinite(A0) or A0 < TN:
        A0 = TN * 1.2
    b0 = min(0.99, max(0.01, T1 / A0)) if A0 > 0 else 0.5

    def loss(par: Sequence[float]) -> float:
        """Sum of squares for a candidate (A, b), guarded against the
        parameter space the models are not defined on."""
        A, b = par
        if not (math.isfinite(A) and math.isfinite(b)) or A <= 0 or not (0 < b < 1):
            return 1e12
        with np.errstate(over="ignore", invalid="ignore"):
            pred = np.asarray(_curve(model, n, A, b), dtype=float)
        if not np.all(np.isfinite(pred)):
            return 1e12
        return float(((pred - y) ** 2).sum())

    fit = optimize.minimize(loss, [A0, b0], method="Nelder-Mead",
                            options={"maxiter": 2000, "xatol": 1e-10,
                                     "fatol": 1e-12})
    A, b = float(fit.x[0]), float(fit.x[1])
    pred = np.asarray(_curve(model, n, A, b), dtype=float)
    return {
        "model": model, "A": A, "b": b,
        # floor(A) as the paper specifies: a fitted A is not an integer, and
        # the number of themes that exist certainly is
        "index": 100 * TN / max(1, math.floor(A)),
        "fitted": [float(v) for v in pred],
        "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
        "reason": "",
    }


def documents_for(fit: Fit, target: float = 95,
                  max_n: int = 1000) -> int | None:
    """How many documents would the fitted curve need to reach ``target``?

    Returns ``None`` when the model does not get there within ``max_n``,
    which is itself worth reporting: a curve that never reaches 95 per cent
    is telling you the material is nowhere near exhausted.
    """
    if not isinstance(fit, dict) or not math.isfinite(fit.get("A", math.nan)):
        return None
    want = target / 100 * math.floor(fit["A"])
    for n in range(1, max_n + 1):
        value = float(_curve(fit["model"], n, fit["A"], fit["b"]))
        if math.isfinite(value) and value >= want:
            return n
    return None
