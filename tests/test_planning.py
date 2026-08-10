"""E36.3 and E36.4, held to the tables in the papers they come from."""

from __future__ import annotations

import math

import pytest

import qdapy

# Donner and Rotondi (2010), Table 2. Their tables used the chi-squared
# critical value rounded to 2.71; with the exact quantile eight of the
# forty-eight cells come out one smaller, which is the correct answer.
TABLE2 = [
    (0.50, 0.40, [559, 373, 301, 255, 264, 146, 112, 95, 228, 120, 89, 76]),
    (0.60, 0.40, [140, 94, 76, 64, 66, 37, 28, 24, 57, 30, 23, 19]),
    (0.70, 0.60, [463, 311, 247, 207, 205, 124, 99, 87, 174, 102, 81, 73]),
    (0.80, 0.60, [116, 78, 62, 52, 52, 31, 25, 22, 44, 26, 21, 19]),
]

# Fugard and Potts (2015), Table 1, at 80 % power.
TABLE1 = {
    0.05: [32, 59, 85, 110, 134, 249, 471, 687],
    0.10: [16, 29, 42, 54, 66, 124, 234, 343],
    0.25: [6, 11, 16, 21, 26, 49, 93, 136],
    0.50: [3, 5, 8, 10, 12, 24, 45, 66],
    0.95: [1, 2, 3, 4, 6, 11, 22, 33],
}


def cells(k0, kl, critical=None):
    return [qdapy.plan_kappa(k0, kl, p, raters=r, critical=critical)
            for p in (0.1, 0.3, 0.5) for r in (2, 3, 4, 5)]


@pytest.mark.parametrize(("k0", "kl", "want"), TABLE2)
def test_the_published_table_is_reproduced_with_its_own_critical_value(k0, kl, want):
    assert cells(k0, kl, critical=2.71) == want


@pytest.mark.parametrize(("k0", "kl", "want"), TABLE2)
def test_the_exact_quantile_is_never_larger_and_never_off_by_more_than_one(k0, kl, want):
    exact = cells(k0, kl)
    assert all(e <= w for e, w in zip(exact, want, strict=True))
    assert all(w - e <= 1 for e, w in zip(exact, want, strict=True))


def test_more_coders_and_a_more_balanced_code_reduce_the_material():
    assert qdapy.plan_kappa(0.8, 0.6, 0.1, raters=2) > qdapy.plan_kappa(0.8, 0.6, 0.1, raters=4)
    assert qdapy.plan_kappa(0.8, 0.6, 0.1) > qdapy.plan_kappa(0.8, 0.6, 0.3)


def test_an_unreachable_bound_is_infinite_rather_than_a_large_number():
    assert qdapy.plan_kappa(0.6, 0.6, 0.3) == math.inf
    assert qdapy.plan_kappa(0.6, 0.7, 0.3) == math.inf


def test_the_lower_bound_and_the_sample_size_are_inverses():
    n = qdapy.plan_kappa(0.8, 0.6, 0.3, raters=3)
    assert qdapy.kappa_lower(n, 0.8, 0.3, raters=3) >= 0.6
    assert qdapy.kappa_lower(n - 1, 0.8, 0.3, raters=3) < 0.6


def test_more_material_buys_a_higher_lower_bound():
    small = qdapy.kappa_lower(30, 0.8, 0.3)
    large = qdapy.kappa_lower(300, 0.8, 0.3)
    assert small < large < 0.8


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        qdapy.plan_kappa(0.8, 0.6, 0, raters=2)
    with pytest.raises(ValueError):
        qdapy.plan_kappa(0.8, 0.6, 0.3, raters=1)


@pytest.mark.parametrize("prevalence", sorted(TABLE1))
def test_fugard_and_potts_table_is_reproduced_exactly(prevalence):
    got = [qdapy.plan_themes(prevalence, instances=k)
           for k in (1, 2, 3, 4, 5, 10, 20, 30)]
    assert got == TABLE1[prevalence]


def test_power_and_sample_size_are_two_views_of_one_calculation():
    n = qdapy.plan_themes(0.05, instances=1, power=0.8)
    assert qdapy.theme_power(n, 0.05) >= 0.8
    assert qdapy.theme_power(n - 1, 0.05) < 0.8


def test_a_rarer_theme_or_more_instances_needs_more_documents():
    assert qdapy.plan_themes(0.05) > qdapy.plan_themes(0.20)
    assert qdapy.plan_themes(0.20, instances=5) > qdapy.plan_themes(0.20)


def test_a_theme_present_in_everyone_needs_one_document():
    assert qdapy.plan_themes(1.0) == 1
    assert qdapy.theme_power(0, 0.5) == 0
