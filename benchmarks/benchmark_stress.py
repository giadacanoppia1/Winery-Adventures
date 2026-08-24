#!/usr/bin/env python3
r"""Script per misurare le prestazioni delle formule di calcolo dello stress.

Cosa fa questo script:
1. Confronta il tempo di esecuzione di diverse versioni dell'algoritmo
   al variare dei dati.
2. Misura quanta memoria RAM viene usata durante l'elaborazione completa.
3. Genera un report finale in formato Markdown.

Metodologia utilizzata (utile per lo studio):
- Warm-up di Numba: le funzioni compilate vengono eseguite a vuoto prima
  di misurarle per non includere il tempo di compilazione.
- Mediana: ogni misurazione è ripetuta più volte. Si usa il valore mediano
  per ignorare eventuali sbalzi di sistema (outlier).
- Limiti: l'algoritmo in Python puro è troppo lento (complessità quadratica)
  e viene saltato per dataset molto grandi.

Uso da terminale:
    python benchmarks/benchmark_stress.py --output docs/performance_report.md
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from winery_adventures.computations import (
    WineryHPCComputations,
    fast_stress_function,
    pairwise_stress_function,
    pairwise_stress_function_parallel,
    pairwise_stress_python,
)
from winery_adventures.transformations import WineryTransformer

# Quantità di dati (numero di rilevazioni) su cui testare gli algoritmi
KERNEL_SIZES: tuple[int, ...] = (100, 500, 1_000, 5_000, 20_000, 100_000)

# Oltre questo numero, non testiamo più la funzione in Python puro
PYTHON_MAX_SIZE: int = 2_000

# Oltre questo numero, saltiamo tutte le funzioni con complessità O(n²)
QUADRATIC_MAX_SIZE: int = 20_000


def make_arrays(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera dati finti per simulare i sensori.

    Restituisce tre array (pH, temperatura e volume).
    Il 'seed' garantisce che i numeri casuali siano sempre gli stessi a
    ogni avvio, così i test sono riproducibili.
    """
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(3.0, 4.0, n),
        rng.uniform(22.0, 28.0, n),
        rng.uniform(200.0, 1000.0, n),
    )


def time_call(func: Callable[..., float], *args, repeats: int = 5) -> float:
    """Misura quanto tempo impiega una funzione a eseguirsi.

    Esegue la funzione più volte (repeats) e calcola la mediana dei tempi.
    Restituisce il tempo in secondi.
    """
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def warm_up_jit() -> None:
    """Riscalda il compilatore JIT (Just-In-Time) di Numba.

    La prima volta che si chiama una funzione con @njit, Numba traduce il
    codice Python in linguaggio macchina. Questa operazione è lenta.
    Eseguendo le funzioni con pochi dati prima dei test veri e propri,
    evitiamo di conteggiare il tempo di compilazione nelle misurazioni.
    """
    tiny = make_arrays(4)
    pairwise_stress_function(*tiny)
    pairwise_stress_function_parallel(*tiny)
    fast_stress_function(*tiny)


def benchmark_kernels(sizes: tuple[int, ...]) -> list[dict]:
    """Testa le singole funzioni di calcolo su dataset di diverse dimensioni.

    Restituisce una lista di dizionari con i tempi di esecuzione.
    """
    records: list[dict] = []

    for n in sizes:
        ph, temp, quantity = make_arrays(n)
        record: dict = {"n": n}

        # Test in Python puro solo se i dati sono pochi
        if n <= PYTHON_MAX_SIZE:
            record["python_s"] = time_call(
                pairwise_stress_python, ph, temp, quantity, repeats=1
            )

        # Test con Numba O(n²) solo se i dati non sono enormi
        if n <= QUADRATIC_MAX_SIZE:
            record["numba_quadratic_s"] = time_call(
                pairwise_stress_function, ph, temp, quantity, repeats=3
            )
            record["numba_parallel_s"] = time_call(
                pairwise_stress_function_parallel, ph, temp, quantity, repeats=3
            )

        # La versione veloce (O(n log n)) viene testata sempre
        record["numba_fast_s"] = time_call(
            fast_stress_function, ph, temp, quantity, repeats=5
        )

        # Controllo di correttezza matematica
        # Verifichiamo che il risultato della versione veloce sia uguale a quello
        # della versione di base, tollerando piccoli errori di arrotondamento.
        if n <= QUADRATIC_MAX_SIZE:
            reference = pairwise_stress_function(ph, temp, quantity)
            fast_value = fast_stress_function(ph, temp, quantity)
            record["relative_error"] = abs(fast_value - reference) / max(reference, 1e-12)

        records.append(record)
        print(f"  n={n:>7,} -> {json.dumps(record, default=float)}")

    return records


def benchmark_pipeline(sensors_path: Path, tank_info_path: Path) -> dict:
    """Misura le prestazioni (tempo e memoria) dell'intero processo sui dati reali."""
    sensors = pl.read_csv(sensors_path, separator="\t")
    tank_info = pl.read_csv(tank_info_path, separator="\t")

    # FASE 1: Misurazione del Tempo
    # Nota: teniamo tracemalloc spento qui perché tracciare la memoria
    # rallenta pesantemente il codice e sfalserebbe i tempi reali.
    start = time.perf_counter()
    transformed = WineryTransformer(tank_info).analyze_data(sensors)
    transform_time = time.perf_counter() - start

    start = time.perf_counter()
    result = WineryHPCComputations().analyze_data(transformed)
    hpc_time = time.perf_counter() - start

    # FASE 2: Misurazione della Memoria
    # Rieseguiamo il calcolo solo per misurare quanta RAM consuma Python.
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()

    WineryHPCComputations().analyze_data(
        WineryTransformer(tank_info).analyze_data(sensors)
    )

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "input_rows": sensors.height,
        "output_rows": result.height,
        "num_tanks": sensors["tank_id"].n_unique(),
        "transform_s": transform_time,
        "hpc_s": hpc_time,
        "peak_python_memory_mb": peak / 1024**2,
        "peak_rss_mb": rss_after / 1024,
        "rss_growth_mb": max(rss_after - rss_before, 0) / 1024,
    }


def render_report(
    kernel_records: list[dict], pipeline_record: dict | None, output: Path
) -> None:
    """Crea e salva un file Markdown con i risultati formattati in tabelle."""

    def fmt(value: float | None, unit: str = "s") -> str:
        """Sceglie l'unità di misura più adatta per leggere bene il tempo.

        Supporta secondi, millisecondi o microsecondi.
        """
        if value is None:
            return "—"
        if not unit:
            return f"{value:.3g}"
        if value < 1e-3:
            return f"{value * 1e6:.1f} µs"
        if value < 1.0:
            return f"{value * 1e3:.2f} ms"
        return f"{value:.3f} s"

    lines = [
        "# Report di performance — formula di stress da fermentazione",
        "",
        "Report generato da `benchmarks/benchmark_stress.py`.",
        "",
        "## Ambiente di esecuzione",
        "",
        f"- Sistema: {platform.platform()}",
        f"- Processore: {platform.processor() or 'n/d'}",
        f"- Python: {platform.python_version()}",
        f"- NumPy: {np.__version__} · Polars: {pl.__version__}",
        "",
        "## Confronto tra implementazioni",
        "",
        "I tempi escludono la compilazione JIT (warm-up preliminare) e sono "
        "mediane su più ripetizioni.",
        "",
        "| n rilevazioni | Python O(n²) | Numba O(n²) | Numba O(n²) parallelo "
        "| Numba O(n log n) | Speed-up vs Python | Errore relativo |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for record in kernel_records:
        python_s = record.get("python_s")
        fast_s = record["numba_fast_s"]
        speedup = f"{python_s / fast_s:,.0f}×" if python_s else "—"
        error = record.get("relative_error")
        lines.append(
            f"| {record['n']:,} | {fmt(python_s)} "
            f"| {fmt(record.get('numba_quadratic_s'))} "
            f"| {fmt(record.get('numba_parallel_s'))} | {fmt(fast_s)} | {speedup} "
            f"| {fmt(error, '')} |"
        )

    lines += [
        "",
        "### Lettura dei risultati",
        "",
        "- **Numba vs Python (stesso algoritmo)**: solo compilando con JIT il "
        "codice diventa molto più veloce. Il calcolo su array vicini in memoria "
        "è l'ideale per i compilatori.",
        "- **Parallelo vs sequenziale**: usare più thread (`prange`) è utile solo "
        "con moltissimi dati. Con pochi dati, attivare i thread richiede più "
        "tempo di quanto ne faccia risparmiare.",
        "- **O(n log n) vs O(n²)**: questo è il vero salto di qualità matematico. "
        "Togliere le doppie sommatorie è essenziale per elaborare oltre 100.000 "
        "righe in tempi accettabili.",
        "- **Errore relativo**: è minuscolo (circa 1e-15), limite dovuto al modo "
        "in cui il computer salva i numeri con la virgola (floating point).",
        "",
    ]

    if pipeline_record:
        lines += [
            "## Pipeline completa sul dataset di produzione",
            "",
            f"- Righe in ingresso: {pipeline_record['input_rows']:,}",
            "- Righe in uscita (esplose per vitigno): "
            f"{pipeline_record['output_rows']:,}",
            f"- Cisterne: {pipeline_record['num_tanks']:,}",
            f"- Tempo trasformazioni: {pipeline_record['transform_s']:.3f} s",
            f"- Tempo computazioni HPC: {pipeline_record['hpc_s']:.3f} s",
            f"- Picco memoria interprete Python "
            f"(`tracemalloc`): {pipeline_record['peak_python_memory_mb']:.2f} MB",
            f"- Picco di memoria reale del processo (`ru_maxrss`): "
            f"{pipeline_record['peak_rss_mb']:.1f} MB",
            f"- Crescita della memoria durante l'esecuzione: "
            f"{pipeline_record['rss_growth_mb']:.1f} MB",
            "",
            "Interpretazione memoria: `tracemalloc` dà un valore molto basso "
            "perché i dati veri non usano oggetti Python (liste o dizionari), "
            "ma sono gestiti in Rust (Polars) e C (NumPy).",
            "",
        ]

    # Crea la cartella di output se non esiste e salva il file
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport scritto in {output}")


def main(argv: list[str] | None = None) -> int:
    """Funzione principale che legge i comandi da terminale e lancia i test."""
    parser = argparse.ArgumentParser(description="Benchmark della formula di stress.")
    parser.add_argument(
        "--output",
        default="docs/performance_report.md",
        help="File Markdown in cui salvare il report.",
    )
    parser.add_argument(
        "--sensors",
        default="data/full_sensors.tsv",
        help="Percorso del dataset dei sensori (per test pipeline).",
    )
    parser.add_argument(
        "--tank-info",
        default="data/full_tank_info.tsv",
        help="Percorso del dataset delle cisterne (per test pipeline).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=max(KERNEL_SIZES),
        help="Numero massimo di rilevazioni per i test.",
    )
    args = parser.parse_args(argv)

    print("Warm-up della compilazione JIT...")
    warm_up_jit()

    print("Benchmark dei calcoli base:")
    sizes = tuple(n for n in KERNEL_SIZES if n <= args.max_size)
    kernel_records = benchmark_kernels(sizes)

    pipeline_record = None
    sensors_path, tank_info_path = Path(args.sensors), Path(args.tank_info)

    if sensors_path.is_file() and tank_info_path.is_file():
        print("\nBenchmark del processo completo:")
        pipeline_record = benchmark_pipeline(sensors_path, tank_info_path)
        print(f"  {json.dumps(pipeline_record, default=float)}")
    else:
        print(
            f"\nDataset {sensors_path} non trovato. Per eseguire il benchmark "
            "completo devi prima generare i dati con `python data_generator.py`."
        )

    render_report(kernel_records, pipeline_record, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
