"""Test suite per le funzioni di calcolo dello stress da fermentazione.

Questi test verificano i casi limite e garantiscono che le implementazioni
ottimizzate producano lo stesso risultato della formula originale.
"""

import math

import numpy as np
import polars as pl
import pytest

from winery_adventures.computations import (
    WineryHPCComputations,
    fast_stress_function,
    pairwise_stress_function,
    pairwise_stress_function_parallel,
    pairwise_stress_python,
)

# Tolleranza per i confronti tra numeri decimali (floating point).
# Le diverse implementazioni sommano i numeri in ordine diverso,
# creando così minime variazioni.
RELATIVE_TOLERANCE = 1e-9


def make_arrays(n: int, seed: int = 0):
    """Genera dati casuali simulando valori reali di una cisterna."""
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(3.0, 4.0, n),
        rng.uniform(22.0, 28.0, n),
        rng.uniform(200.0, 1000.0, n),
    )


@pytest.mark.parametrize(
    "implementation",
    [
        pairwise_stress_python,
        pairwise_stress_function,
        pairwise_stress_function_parallel,
        fast_stress_function,
    ],
)
def test_empty_input_returns_zero(implementation):
    """Se non ci sono rilevazioni, il calcolo deve restituire 0.0."""
    empty = np.empty(0, dtype=np.float64)
    assert implementation(empty, empty, empty) == 0.0


@pytest.mark.parametrize(
    "implementation",
    [
        pairwise_stress_function,
        pairwise_stress_function_parallel,
        fast_stress_function,
    ],
)
def test_single_reading_has_no_stress(implementation):
    """Con una sola rilevazione lo stress deve essere 0.0."""
    values = np.array([3.5]), np.array([25.0]), np.array([500.0])
    assert implementation(*values) == 0.0


@pytest.mark.parametrize(
    "implementation",
    [
        pairwise_stress_function,
        pairwise_stress_function_parallel,
        fast_stress_function,
    ],
)
def test_identical_readings_have_no_stress(implementation):
    """Se tutte le rilevazioni sono identiche, lo stress è 0.0."""
    n = 25
    values = (np.full(n, 3.5), np.full(n, 25.0), np.full(n, 750.0))
    assert implementation(*values) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n", [2, 7, 64, 257, 1000])
def test_implementations_agree(n):
    """Verifica che le versioni ottimizzate concordino con quella base."""
    arrays = make_arrays(n, seed=n)
    expected = pairwise_stress_python(*arrays)

    assert math.isclose(
        pairwise_stress_function(*arrays),
        expected,
        rel_tol=RELATIVE_TOLERANCE,
    )
    assert math.isclose(
        pairwise_stress_function_parallel(*arrays),
        expected,
        rel_tol=RELATIVE_TOLERANCE,
    )
    assert math.isclose(
        fast_stress_function(*arrays),
        expected,
        rel_tol=RELATIVE_TOLERANCE,
    )


def test_smaller_tanks_are_more_stressed():
    """Verifica che una cisterna più piccola subisca uno stress maggiore."""
    ph = np.array([3.3, 3.7])
    temp = np.array([24.0, 27.0])

    small = fast_stress_function(ph, temp, np.array([300.0, 300.0]))
    large = fast_stress_function(ph, temp, np.array([1500.0, 1500.0]))

    assert small > large


def test_stress_is_invariant_to_reading_order():
    """L'ordine dei dati non deve alterare il risultato finale."""
    ph, temp, quantity = make_arrays(200, seed=7)
    permutation = np.random.default_rng(1).permutation(ph.size)

    original = fast_stress_function(ph, temp, quantity)
    shuffled = fast_stress_function(
        ph[permutation], temp[permutation], quantity[permutation]
    )

    assert math.isclose(original, shuffled, rel_tol=RELATIVE_TOLERANCE)


def test_missing_columns_raise_value_error():
    """Assicura che venga sollevato un errore se mancano le colonne."""
    df = pl.DataFrame({"tank_id": [1], "pH": [3.4]})

    with pytest.raises(ValueError, match="colonne"):
        WineryHPCComputations().analyze_data(df)


def test_null_and_invalid_rows_are_excluded_but_kept():
    """Le righe non valide vengono ignorate ma mantenute in output."""
    df = pl.DataFrame(
        {
            "tank_id": [1, 1, 1],
            "pH": [3.4, 3.6, None],
            "temp": [25.0, 26.0, 24.0],
            "quantity_liters": [500, 500, 0],
        }
    )

    out = WineryHPCComputations().analyze_data(df)

    assert out.height == 3, "Il DataFrame in output deve avere lo stesso numero di righe."
    assert out["stress_score"][0] == pytest.approx(2.2)


def test_tank_without_valid_readings_gets_zero():
    """Cisterne senza dati validi ricevono uno stress pari a 0.0."""
    df = pl.DataFrame(
        {
            "tank_id": [1, 2, 2],
            "pH": [None, 3.4, 3.6],
            "temp": [25.0, 25.0, 26.0],
            "quantity_liters": [500, 500, 500],
        }
    )

    out = WineryHPCComputations().analyze_data(df)

    assert out.filter(pl.col("tank_id") == 1)["stress_score"][0] == 0.0
    assert out.filter(pl.col("tank_id") == 2)["stress_score"][0] == pytest.approx(2.2)


def test_fast_and_quadratic_paths_agree_on_dataframe():
    """Verifica la coerenza tra algoritmo O(n^2) e O(n log n)."""
    rng = np.random.default_rng(3)
    size = 300
    df = pl.DataFrame(
        {
            "tank_id": [1] * size,
            "pH": rng.uniform(3.0, 4.0, size),
            "temp": rng.uniform(22.0, 28.0, size),
            "quantity_liters": rng.uniform(200.0, 1000.0, size),
        }
    )

    fast = WineryHPCComputations(use_fast_algorithm=True).analyze_data(df)
    exact = WineryHPCComputations(use_fast_algorithm=False).analyze_data(df)

    assert math.isclose(
        fast["stress_score"][0],
        exact["stress_score"][0],
        rel_tol=RELATIVE_TOLERANCE,
    )


def test_integer_columns_are_accepted():
    """Assicura che colonne intere vengano castate in automatico."""
    df = pl.DataFrame(
        {
            "tank_id": [1, 1],
            "pH": [3.4, 3.6],
            "temp": [25, 26],
            "quantity_liters": [500, 500],
        }
    )

    assert WineryHPCComputations().analyze_data(df)["stress_score"][0] == 2.2
