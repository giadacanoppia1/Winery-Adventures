"""Calcolo dello stress di fermentazione.

Contiene quattro implementazioni equivalenti con diverse ottimizzazioni:
1. pairwise_stress_python: Python puro (lento, O(n^2)), utile solo per test.
2. pairwise_stress_function: Ottimizzato con Numba JIT (O(n^2)).
3. pairwise_stress_function_parallel: Come il precedente, ma parallelizzato.
4. fast_stress_function: Riscrittura matematica super-veloce (O(n log n)).
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
    """Implementazione base in Python puro.

    Da usare solo per test e confronti.
    """
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
    """Versione ottimizzata con Numba.

    - nogil=True: permette l'uso del multithreading reale (ignora il GIL).
    - cache=True: salva la funzione compilata per avvii successivi più rapidi.
    """
    n = pH_vals.shape[0]
    if n == 0:
        return 0.0

    stress_sum = 0.0
    for i in range(n):
        # Ottimizzazione: calcoliamo il termine costante per 'i' una volta sola
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
    """Versione parallela con Numba (tramite prange nel ciclo esterno).

    Ideale quando si analizza una singola cisterna con una mole enorme di dati.
    """
    n = pH_vals.shape[0]
    if n == 0:
        return 0.0

    partial = np.zeros(n, dtype=np.float64)
    # prange divide automaticamente il carico di lavoro tra i vari core della CPU
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
    """Calcola la somma delle differenze assolute in modo efficiente: O(n log n).

    Lo fa ordinando prima i valori e utilizzando la tecnica delle somme prefisse,
    evitando così di dover calcolare la differenza per ogni singola coppia.
    """
    n = values.shape[0]
    result = np.zeros(n, dtype=np.float64)
    if n < 2:
        return result

    # Ordiniamo gli elementi per poter applicare il metodo delle somme prefisse
    order = np.argsort(values)
    prefix = np.zeros(n + 1, dtype=np.float64)

    for k in range(n):
        prefix[k + 1] = prefix[k] + values[order[k]]
    total = prefix[n]

    # Calcoliamo le differenze sfruttando l'ordinamento
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
    """Implementazione matematica ottimizzata (O(n log n)).

    Sfrutta le proprietà algebriche della formula originale per scomporre il
    problema, appoggiandosi a _sum_abs_differences per eseguire il calcolo.
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
    """Sceglie dinamicamente l'algoritmo (O(n^2) o O(n log n))."""
    if not use_fast_algorithm or pH_vals.shape[0] <= pairwise_threshold:
        return pairwise_stress_function(pH_vals, temp_vals, quantity_vals)
    return fast_stress_function(pH_vals, temp_vals, quantity_vals)


class WineryHPCComputations(BaseWineryAnalyzer):
    """Classe per orchestrare il calcolo dello stress in parallelo."""

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "tank_id",
        "pH",
        "temp",
        "quantity_liters",
    )
    OUTPUT_COLUMN: str = "stress_score"

    def __init__(
        self,
        n_jobs: int = -1,
        use_fast_algorithm: bool = True,
        pairwise_threshold: int = 64,
    ) -> None:
        """Configura le impostazioni dei worker e i criteri algoritmici."""
        self.n_jobs = n_jobs
        self.use_fast_algorithm = use_fast_algorithm
        self.pairwise_threshold = pairwise_threshold

    def analyze_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calcola e aggiunge la colonna 'stress_score' per ogni cisterna.

        Ignora automaticamente le righe con dati non validi (nulli o volume <= 0).
        """
        missing = [column for column in self.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(
                "Impossibile calcolare lo stress di fermentazione: "
                f"colonne mancanti {sorted(missing)}. "
                f"Colonne disponibili: {df.columns}"
            )

        # 1. Filtriamo per mantenere solo i record validi
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

        # 2. Prepariamo i task suddividendo il DataFrame cisterna per cisterna
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

        # 3. Lanciamo l'elaborazione
        scores = self._execute(tasks)

        # 4. Creiamo un nuovo DataFrame con i risultati ottenuti
        stress_df = pl.DataFrame(
            {
                "tank_id": pl.Series(tank_ids, dtype=df.schema["tank_id"]),
                self.OUTPUT_COLUMN: pl.Series(scores, dtype=pl.Float64),
            }
        )

        # 5. Effettuiamo una left join per riportare i risultati nell'originale
        return df.join(stress_df, on="tank_id", how="left").with_columns(
            pl.col(self.OUTPUT_COLUMN).fill_null(0.0)
        )

    def _execute(self, tasks: Sequence[tuple]) -> list[float]:
        """Esegue i task in parallelo tramite joblib e i thread Python."""
        if not tasks:
            return []

        results: Iterable[float] | None = joblib.Parallel(
            n_jobs=self.n_jobs, prefer="threads"
        )(tasks)

        if results is None:
            # Piano B: fallback sequenziale in caso di problemi col parallelismo
            logger.debug("Backend joblib senza risultati: fallback sequenziale.")
            results = [func(*args, **kwargs) for func, args, kwargs in tasks]

        return list(results)
