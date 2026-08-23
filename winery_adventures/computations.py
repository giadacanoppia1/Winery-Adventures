"""Formula di stress da fermentazione: quattro implementazioni equivalenti.

stress = (1/n^2) * sum_i sum_j (|pH_i-pH_j| + 2|T_i-T_j|) * (500/q_i + 500/q_j)

pairwise_stress_python        -> O(n^2), Python puro, oracolo di correttezza
pairwise_stress_function      -> O(n^2), compilata JIT con Numba
pairwise_stress_function_parallel -> come sopra, parallela con prange
fast_stress_function          -> O(n log n), riscrittura algebrica, usata in produzione
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import joblib
import numpy as np
import polars as pl
from numba import njit, prange

from winery_adventures.base import BaseWineryAnalyzer

__all__ = [
    "WineryHPCComputations",
    "pairwise_stress_function",
    "pairwise_stress_function_parallel",
    "fast_stress_function",
    "pairwise_stress_python",
]

logger = logging.getLogger(__name__)

VOLUME_CONSTANT: float = 500.0
TEMPERATURE_WEIGHT: float = 2.0


def pairwise_stress_python(
    pH_vals: np.ndarray, temp_vals: np.ndarray, quantity_vals: np.ndarray
) -> float:
    """Traduzione letterale della formula in Python puro. Solo per benchmark/test."""
    n = len(pH_vals)
    if n == 0:
        return 0.0

    stress_sum = 0.0
    for i in range(n):
        for j in range(n):
            ph_dev = abs(pH_vals[i] - pH_vals[j])
            t_dev = abs(temp_vals[i] - temp_vals[j]) * TEMPERATURE_WEIGHT
            quantity_factor = (
                VOLUME_CONSTANT / quantity_vals[i] + VOLUME_CONSTANT / quantity_vals[j]
            )
            stress_sum += (ph_dev + t_dev) * quantity_factor

    return stress_sum / (n * n)


@njit(cache=True, nogil=True)
def pairwise_stress_function(
    pH_vals: np.ndarray, temp_vals: np.ndarray, quantity_vals: np.ndarray
) -> float:
    """Stessa formula di ``pairwise_stress_python``, compilata JIT con Numba.

    nogil=True: rilascia il GIL, necessario per parallelizzare con thread joblib.
    cache=True: evita di ricompilare a ogni avvio.
    """
    n = pH_vals.shape[0]
    if n == 0:
        return 0.0

    stress_sum = 0.0
    for i in range(n):
        # 500/q_i calcolato una sola volta per riga, non n volte.
        inv_qi = VOLUME_CONSTANT / quantity_vals[i]
        ph_i = pH_vals[i]
        temp_i = temp_vals[i]
        row_sum = 0.0
        for j in range(n):
            ph_dev = abs(ph_i - pH_vals[j])
            t_dev = abs(temp_i - temp_vals[j]) * TEMPERATURE_WEIGHT
            quantity_factor = inv_qi + VOLUME_CONSTANT / quantity_vals[j]
            row_sum += (ph_dev + t_dev) * quantity_factor
        stress_sum += row_sum

    return stress_sum / (n * n)


@njit(cache=True, nogil=True, parallel=True)
def pairwise_stress_function_parallel(
    pH_vals: np.ndarray, temp_vals: np.ndarray, quantity_vals: np.ndarray
) -> float:
    """Come ``pairwise_stress_function``, con il ciclo esterno su prange.

    Utile quando una singola cisterna ha moltissime rilevazioni.
    """
    n = pH_vals.shape[0]
    if n == 0:
        return 0.0

    partial = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        inv_qi = VOLUME_CONSTANT / quantity_vals[i]
        ph_i = pH_vals[i]
        temp_i = temp_vals[i]
        row_sum = 0.0
        for j in range(n):
            ph_dev = abs(ph_i - pH_vals[j])
            t_dev = abs(temp_i - temp_vals[j]) * TEMPERATURE_WEIGHT
            row_sum += (ph_dev + t_dev) * (inv_qi + VOLUME_CONSTANT / quantity_vals[j])
        partial[i] = row_sum

    return partial.sum() / (n * n)


@njit(cache=True, nogil=True)
def _sum_abs_differences(values: np.ndarray) -> np.ndarray:
    """S[i] = somma_j |values[i] - values[j]|, in O(n log n) tramite somme prefisse.

    Su valori ordinati, per l'elemento in posizione k tutti i precedenti sono
    minori e tutti i successivi maggiori: il valore assoluto si elimina e la
    somma si riduce a differenze di somme prefisse.
    """
    n = values.shape[0]
    result = np.zeros(n, dtype=np.float64)
    if n < 2:
        return result

    order = np.argsort(values)

    prefix = np.zeros(n + 1, dtype=np.float64)
    for k in range(n):
        prefix[k + 1] = prefix[k] + values[order[k]]
    total = prefix[n]

    for k in range(n):
        x = values[order[k]]
        left = k * x - prefix[k]
        right = (total - prefix[k + 1]) - (n - 1 - k) * x
        result[order[k]] = left + right

    return result


@njit(cache=True, nogil=True)
def fast_stress_function(
    pH_vals: np.ndarray, temp_vals: np.ndarray, quantity_vals: np.ndarray
) -> float:
    """Formula di stress in O(n log n), algebricamente equivalente all'originale.

    Con w_i = 500/q_i e D_ij = |pH_i-pH_j| + 2|T_i-T_j| (simmetrico in i, j):

        sum_ij D_ij*(w_i+w_j) = 2 * sum_i w_i * (sum_j D_ij)

    perché scambiando i nomi degli indici i due addendi coincidono. La somma
    interna sum_j D_ij si ottiene da _sum_abs_differences in O(n log n)
    invece di O(n).
    """
    n = pH_vals.shape[0]
    if n == 0:
        return 0.0

    ph_sums = _sum_abs_differences(pH_vals)
    temp_sums = _sum_abs_differences(temp_vals)

    stress_sum = 0.0
    for i in range(n):
        row_total = ph_sums[i] + TEMPERATURE_WEIGHT * temp_sums[i]
        stress_sum += (VOLUME_CONSTANT / quantity_vals[i]) * row_total

    return 2.0 * stress_sum / (n * n)


def _compute_tank_stress(
    pH_vals: np.ndarray,
    temp_vals: np.ndarray,
    quantity_vals: np.ndarray,
    use_fast_algorithm: bool,
    pairwise_threshold: int,
) -> float:
    """Sceglie O(n^2) o O(n log n) in base alla numerosità della cisterna.

    Funzione di modulo, non lambda: deve restare serializzabile da joblib.
    """
    if not use_fast_algorithm or pH_vals.shape[0] <= pairwise_threshold:
        return pairwise_stress_function(pH_vals, temp_vals, quantity_vals)
    return fast_stress_function(pH_vals, temp_vals, quantity_vals)


class WineryHPCComputations(BaseWineryAnalyzer):
    """Calcola lo stress di ogni cisterna, in parallelo con joblib su thread."""

    REQUIRED_COLUMNS: tuple[str, ...] = ("tank_id", "pH", "temp", "quantity_liters")
    OUTPUT_COLUMN: str = "stress_score"

    def __init__(
        self,
        n_jobs: int = -1,
        use_fast_algorithm: bool = True,
        pairwise_threshold: int = 64,
    ) -> None:
        """Configura numero di worker e strategia dell'algoritmo."""
        self.n_jobs = n_jobs
        self.use_fast_algorithm = use_fast_algorithm
        self.pairwise_threshold = pairwise_threshold

    def analyze_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggiunge la colonna ``stress_score``, calcolata cisterna per cisterna.

        Righe con pH/temp/volume nulli o volume <= 0 sono escluse dal calcolo
        ma restano nel DataFrame. Cisterne senza rilevazioni valide ricevono
        stress 0.0.
        """
        missing = [column for column in self.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(
                "Impossibile calcolare lo stress di fermentazione: colonne "
                f"mancanti {sorted(missing)}. Colonne disponibili: {df.columns}"
            )

        valid = df.filter(
            pl.col("pH").is_not_null()
            & pl.col("temp").is_not_null()
            & pl.col("quantity_liters").is_not_null()
            & (pl.col("quantity_liters") > 0)
        )
        discarded = df.height - valid.height
        if discarded:
            logger.warning(
                "%d righe su %d escluse dal calcolo dello stress "
                "(valori mancanti o volume non positivo).",
                discarded,
                df.height,
            )

        tank_ids: list[Any] = []
        tasks: list[tuple[Callable[..., float], tuple, dict]] = []

        # Solo le colonne necessarie: partition_by restituisce array NumPy
        # senza passare per liste Python.
        for keys, group in (
            valid.select(self.REQUIRED_COLUMNS)
            .partition_by("tank_id", as_dict=True, maintain_order=True)
            .items()
        ):
            tank_ids.append(keys[0])
            tasks.append(
                joblib.delayed(_compute_tank_stress)(
                    group["pH"].cast(pl.Float64).to_numpy(),
                    group["temp"].cast(pl.Float64).to_numpy(),
                    group["quantity_liters"].cast(pl.Float64).to_numpy(),
                    self.use_fast_algorithm,
                    self.pairwise_threshold,
                )
            )

        logger.info("Calcolo dello stress su %d cisterne.", len(tasks))
        scores = self._execute(tasks)

        stress_df = pl.DataFrame(
            {
                "tank_id": pl.Series(tank_ids, dtype=df.schema["tank_id"]),
                self.OUTPUT_COLUMN: pl.Series(scores, dtype=pl.Float64),
            }
        )

        return df.join(stress_df, on="tank_id", how="left").with_columns(
            pl.col(self.OUTPUT_COLUMN).fill_null(0.0)
        )

    def _execute(self, tasks: Sequence[tuple]) -> list[float]:
        """Esegue i task con joblib su thread (i kernel Numba rilasciano il GIL)."""
        if not tasks:
            return []

        results: Iterable[float] | None = joblib.Parallel(
            n_jobs=self.n_jobs, prefer="threads"
        )(tasks)

        if results is None:
            # Fallback sequenziale: lento ma non lascia la pipeline senza dati.
            logger.debug("Backend joblib senza risultati: fallback sequenziale.")
            results = [func(*args, **kwargs) for func, args, kwargs in tasks]

        return list(results)
