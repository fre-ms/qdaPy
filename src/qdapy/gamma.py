"""Gamma: agreement measured with respect to the best alignment.

Mathet, Widlocher and Metivier (2015) doi:10.1162/coli_a_00227.

Every other coefficient in this package fixes the alignment first and then
measures agreement on it. Krippendorff's unitizing alpha compares units that
overlap; WindowDiff and Pk compare boundaries. Gamma refuses that separation:
it searches for the pairing of units across annotators that minimises a
combined positional and categorical disorder, and reports agreement with
respect to that pairing. Unitizing and categorisation are judged together --
"unified and holistic", as the title has it.

The practical difference is that gamma can pair two units that do not overlap
at all, when the surrounding configuration says they refer to the same
phenomenon. Alpha cannot express that.

The cost is real: finding the best alignment is a set-partitioning problem,
NP-hard for three or more annotators. This module solves it exactly, with the
paper's pruning theorem and an admissible bound, and refuses when the search
would exceed a budget. A gamma produced by a heuristic is not gamma.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

from .unitizing import Segment

__all__ = [
    "EMPTY",
    "best_alignment",
    "dissimilarity",
    "gamma",
    "unitary_disorder",
]

# Delta_empty: a constant of the method, not a parameter (section 4.4.1).
EMPTY = 1.0

# Gamma consumes exactly what qdapy.segments() produces, so it is the same
# type as the unitizing alpha's input rather than a look-alike of its own.
Unit = Segment

# The category distance of equation (4): two values in, a number in [0, 1].
CategoryDistance = Callable[[object, object], float]


class Alignment(TypedDict, total=False):
    """What :func:`best_alignment` reports.

    Two shapes, and ``exhausted`` says which one you have.  On success:
    ``disorder``, ``raw_disorder``, ``alignment``, ``mean_units``, ``nodes``,
    ``candidates``, ``exhausted=False``.  When the node budget ran out:
    ``exhausted=True``, ``nodes`` and ``candidates`` only -- there is no
    alignment to report, and reporting the best found so far would be a
    guess dressed as a result.

    Total is ``False`` because that is the truth about the mapping; check
    ``exhausted`` before reading anything else.
    """

    disorder: float
    raw_disorder: float
    alignment: list[list[str]]
    mean_units: float
    nodes: int
    candidates: int
    exhausted: bool


class GammaResult(TypedDict, total=False):
    """What :func:`gamma` reports.

    On success: ``gamma``, ``observed``, ``expected``, ``expected_sd``,
    ``samples``, ``nodes``.  When the coefficient could not be had, ``gamma``
    is NaN and ``reason`` says why in words -- either the alignment search
    exceeded its budget, or the expected disorder came out as zero.
    """

    gamma: float
    observed: float
    expected: float
    expected_sd: float
    samples: int
    recommended_samples: int
    alignment: list[list[str]]
    nodes: int
    candidates: int
    exhausted: bool
    reason: str


class _Search(TypedDict):
    """Mutable state of the branch-and-bound run in :func:`best_alignment`."""

    best: float
    set: list[int] | None
    nodes: int
    exhausted: bool


def _pos(u: Unit, v: Unit) -> float:
    """Equation (3). Scale-free, and squared so it grows faster with drift."""
    span = (u["end"] - u["start"]) + (v["end"] - v["start"])
    if span <= 0:
        return EMPTY
    r = (abs(u["start"] - v["start"]) + abs(u["end"] - v["end"])) / span
    return r * r * EMPTY


def _cat(u: Unit, v: Unit, dist_cat: CategoryDistance | None) -> float:
    """Equation (4), from a category distance in [0, 1]."""
    d = (dist_cat(u["value"], v["value"]) if dist_cat
         else (0.0 if str(u["value"]) == str(v["value"]) else 1.0))
    return max(0.0, min(1.0, d)) * EMPTY


def dissimilarity(u: Unit | None, v: Unit | None, *,
                  dist_cat: CategoryDistance | None = None,
                  alpha: float = 1, beta: float = 1) -> float:
    """Equation (5) with both weights at one, which is what gamma uses.

    A unit in the right place with the wrong code therefore costs the same as
    one with the right code in a badly wrong place. ``None`` stands for the
    empty unit and costs ``EMPTY`` against anything, including another empty.
    """
    if u is None or v is None:
        return EMPTY
    return alpha * _pos(u, v) + beta * _cat(u, v, dist_cat)


def unitary_disorder(tuple_: Sequence[Unit | None], **kw: Any) -> float:
    """Equation (6): the average over all C(n,2) annotator pairs.

    Averaging rather than summing keeps the value independent of the number
    of annotators. A unitary alignment holding one real unit costs exactly
    ``EMPTY``, because every pair in it involves an empty.
    """
    n = len(tuple_)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            total += dissimilarity(tuple_[i], tuple_[j], **kw)
            pairs += 1
    return total / pairs


def _candidates(
    by_coder: Sequence[Sequence[Unit]], **kw: Any,
) -> list[tuple[tuple[str, ...], float]]:
    """Every combination of at most one unit per annotator, at least one real.

    Equation (9) lets us discard any whose disorder exceeds n * EMPTY: such a
    combination can always be replaced by separate single-unit alignments, so
    it cannot belong to the best alignment. The paper reports this removes
    about 90 per cent of them.
    """
    n = len(by_coder)
    limit = n * EMPTY + 1e-12
    out = []

    def build(coder: int, tuple_: list[Unit | None],
              members: list[str]) -> None:
        """Extend the tuple one annotator at a time, depth first.

        Each annotator contributes one of its units or nothing; a tuple of
        nothing but empties is not an alignment and is dropped.
        """
        if coder == n:
            if not members:
                return
            d = unitary_disorder(tuple_, **kw)
            if d <= limit:
                out.append((tuple(members), d))
            return
        build(coder + 1, [*tuple_, None], members)
        for k, unit in enumerate(by_coder[coder]):
            build(coder + 1, [*tuple_, unit], [*members, f"{coder}:{k}"])

    build(0, [], [])
    return out


def best_alignment(  # noqa: C901 -- branch and bound; the branches are the algorithm
    by_coder: Sequence[Sequence[Unit]],
    *,
    max_nodes: int = 200000,
    **kw: Any,
) -> Alignment | None:
    """The alignment minimising the total disorder, found exactly.

    Depth-first over the lowest-numbered uncovered unit, with an admissible
    bound: every remaining unit must land in some candidate, and a candidate
    covers at most n units, so the cheapest possible remainder is the sum of
    the units' cheapest containing candidates divided by n.

    Returns ``{"exhausted": True, ...}`` rather than a guess when the budget
    runs out.
    """
    n = len(by_coder)
    ids = [f"{c}:{k}" for c, units in enumerate(by_coder)
           for k in range(len(units))]
    if not ids:
        return None
    index = {u: i for i, u in enumerate(ids)}

    cands = _candidates(by_coder, **kw)
    by_unit: list[list[int]] = [[] for _ in ids]
    for ci, (members, _) in enumerate(cands):
        for m in members:
            by_unit[index[m]].append(ci)
    if any(not lst for lst in by_unit):
        return None
    cheapest = [min(cands[ci][1] for ci in lst) for lst in by_unit]

    covered = [False] * len(ids)
    state: _Search = {"best": math.inf, "set": None,
                      "nodes": 0, "exhausted": False}

    def search(chosen: list[int], cost: float) -> None:  # noqa: C901 -- the recursive step
        """Branch on the lowest-numbered uncovered unit; bound and prune.

        Records the cheapest complete cover it reaches in ``state``. Gives
        up when the node budget is spent, setting ``exhausted`` rather than
        returning the best found so far -- see the module docstring on why
        an approximate gamma is not gamma.
        """
        if state["exhausted"]:
            return
        state["nodes"] += 1
        if state["nodes"] > max_nodes:
            state["exhausted"] = True
            return
        first = next((i for i, c in enumerate(covered) if not c), None)
        if first is None:
            if cost < state["best"]:
                state["best"], state["set"] = cost, list(chosen)
            return
        bound = sum(cheapest[i] for i, c in enumerate(covered) if not c) / n
        if cost + bound >= state["best"]:
            return
        for ci in sorted(by_unit[first], key=lambda c: cands[c][1]):
            members = cands[ci][0]
            if any(covered[index[m]] for m in members):
                continue
            for m in members:
                covered[index[m]] = True
            chosen.append(ci)
            search(chosen, cost + cands[ci][1])
            chosen.pop()
            for m in members:
                covered[index[m]] = False
            if state["exhausted"]:
                return

    search([], 0.0)
    if state["exhausted"] or state["set"] is None:
        return {"exhausted": True, "nodes": state["nodes"],
                "candidates": len(cands)}
    mean_units = len(ids) / n
    return {
        "disorder": state["best"] / mean_units,
        "raw_disorder": state["best"],
        "alignment": [list(cands[ci][0]) for ci in state["set"]],
        "mean_units": mean_units, "nodes": state["nodes"],
        "candidates": len(cands), "exhausted": False,
    }


def _shift(units: Sequence[Unit], length: float, offset: float) -> list[Unit]:
    """The circular shift of section 5.2.1.

    Split the annotator's units at a random position and swap the parts:
    every unit keeps its length and its category, and only the alignment
    between annotators is destroyed.
    """
    out: list[Unit] = []
    for u in units:
        start = (u["start"] + offset) % length
        out.append({"start": start,
                    "end": start + (u["end"] - u["start"]),
                    "value": u["value"]})
    return sorted(out, key=lambda u: u["start"])


def gamma(by_coder: Sequence[Sequence[Unit]], *,
          dist_cat: CategoryDistance | None = None,
          alpha: float = 1, beta: float = 1, samples: int = 30,
          seed: int = 42, max_nodes: int = 200000) -> GammaResult | None:
    """Equation (8): 1 minus observed disorder over expected disorder.

    Chance correction is by sampling, because the random variable is a whole
    annotation rather than a pair of judgements and its expected value cannot
    be had analytically. The generator is the plugin's, seeded, so qdaZ, qdaR
    and qdaPy report the same expected value for the same data.

    ``recommended_samples`` applies the paper's sampling rule (section 5.3):
    how many random continua the observed variability suggests for two per
    cent precision. If it exceeds ``samples``, run it again with more.
    """
    from .reliability import mulberry32

    kw = {"dist_cat": dist_cat, "alpha": alpha, "beta": beta}
    coders = [sorted((u for u in (units or [])
                      if math.isfinite(u["start"]) and math.isfinite(u["end"])
                      and u["end"] > u["start"]), key=lambda u: u["start"])
              for units in by_coder]
    if len(coders) < 2 or any(not c for c in coders):
        return None

    observed = best_alignment(coders, max_nodes=max_nodes, **kw)
    if observed is None or observed.get("exhausted"):
        return {"gamma": math.nan, "exhausted": True,
                "candidates": observed["candidates"] if observed else 0,
                "reason": "the search for the best alignment exceeded "
                          "max_nodes; gamma is exact or it is nothing"}

    span = max(u["end"] for units in coders for u in units)
    rand = mulberry32(seed)
    values = []
    for _ in range(samples):
        offsets = [int(rand() * span) for _ in coders]
        shifted = [_shift(units, span, off)
                   for units, off in zip(coders, offsets, strict=True)]
        r = best_alignment(shifted, max_nodes=max_nodes, **kw)
        if r is not None and not r.get("exhausted"):
            values.append(r["disorder"])
    if not values or sum(values) / len(values) <= 0:
        return {"gamma": math.nan, "observed": observed["disorder"],
                "expected": math.nan,
                "reason": "the expected disorder came out as zero"}

    mean = sum(values) / len(values)
    sd = (math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
          if len(values) > 1 else 0.0)
    return {
        "gamma": 1 - observed["disorder"] / mean,
        "observed": observed["disorder"], "expected": mean, "expected_sd": sd,
        "samples": len(values),
        "recommended_samples": (math.ceil(((sd / mean) * 1.959963984540054
                                           / 0.02) ** 2) if mean > 0 else 0),
        "alignment": observed["alignment"], "candidates": observed["candidates"],
        "exhausted": False, "reason": "",
    }
