#!/usr/bin/env python3
r"""Script per misurare le prestazioni delle formule di calcolo dello stress.

Cosa fa questo script:
1. Confronta il tempo di esecuzione di diverse versioni dell'algoritmo.
2. Misura quanta memoria RAM viene usata durante l'elaborazione.
3. Genera un report finale in formato Markdown.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
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

# Gestione cross-platform per il modulo 'resource' (nativo Unix/Linux/macOS)
try:
    import resource
except ImportError:
    resource = None

# Quantità di dati (numero di rilevazioni) su cui testare gli algoritmi
KERNEL_SIZES: tuple[int, ...] = (100, 500, 1_000, 5_000, 20_000, 100_000)

# Oltre questo numero, non testiamo più la funzione in Python puro
PYTHON_MAX_SIZE: int = 2_000

# Oltre questo numero, saltiamo tutte le funzioni con complessità O(n²)
QUADRATIC_MAX_SIZE: int = 20_000


def make_arrays(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera dati casuali per simulare i sensori."""
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(3.0, 4.0, n),
        rng.uniform(22.0, 28.0, n),
        rng.uniform(200.0, 1000.0, n),
    )


def time_call(func: Callable[..., float], *args, repeats: int = 5) -> float:
    """Misura quanto tempo impiega una funzione a eseguirsi (mediana)."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def warm_up_jit() -> None:
    """Riscalda il compilatore JIT di Numba per evitare di conteggiare i tempi."""
    tiny = make_arrays(4)
    pairwise_stress_function(*tiny)
    pairwise_stress_function_parallel(*tiny)
    fast_stress_function(*tiny)


def benchmark_kernels(sizes: tuple[int, ...]) -> list[dict]:
    """Testa le singole funzioni di calcolo su dataset di diverse dimensioni."""
    records: list[dict] = []

    for n in sizes:
        ph, temp, quantity = make_arrays(n)
        record: dict = {"n": n}

        if n <= PYTHON_MAX_SIZE:
            record["python_s"] = time_call(
                pairwise_stress_python, ph, temp, quantity, repeats=1
            )

        if n <= QUADRATIC_MAX_SIZE:
            record["numba_quadratic_s"] = time_call(
                pairwise_stress_function, ph, temp, quantity, repeats=3
            )
            record["numba_parallel_s"] = time_call(
                pairwise_stress_function_parallel,
                ph,
                temp,
                quantity,
                repeats=3,
            )

        record["numba_fast_s"] = time_call(
            fast_stress_function, ph, temp, quantity, repeats=5
        )

        if n <= QUADRATIC_MAX_SIZE:
            reference = pairwise_stress_function(ph, temp, quantity)
            fast_value = fast_stress_function(ph, temp, quantity)
            record["relative_error"] = abs(fast_value - reference) / max(reference, 1e-12)

        records.append(record)
        print(f"  n={n:>7,} -> {json.dumps(record, default=float)}")

    return records


def get_rss_memory() -> float:
    """Restituisce l'occupazione RSS in megabyte (compatibile Windows/Linux)."""
    if resource is not None:
        # Su Linux/macOS
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    else:
        # Su Windows tramite psutil (se installato)
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0


def benchmark_pipeline(sensors_path: Path, tank_info_path: Path) -> dict:
    """Misura le prestazioni (tempo e memoria) dell'intero processo."""
    sensors = pl.read_csv(sensors_path, separator="\t")
    tank_info = pl.read_csv(tank_info_path, separator="\t")

    start = time.perf_counter()
    transformed = WineryTransformer(tank_info).analyze_data(sensors)
    transform_time = time.perf_counter() - start

    start = time.perf_counter()
    result = WineryHPCComputations().analyze_data(transformed)
    hpc_time = time.perf_counter() - start

    rss_before = get_rss_memory()
    tracemalloc.start()

    WineryHPCComputations().analyze_data(
        WineryTransformer(tank_info).analyze_data(sensors)
    )

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = get_rss_memory()

    return {
        "input_rows": sensors.height,
        "output_rows": result.height,
        "num_tanks": sensors["tank_id"].n_unique(),
        "transform_s": transform_time,
        "hpc_s": hpc_time,
        "peak_python_memory_mb": peak / 1024**2,
        "peak_rss_mb": rss_after,
        "rss_growth_mb": max(rss_after - rss_before, 0.0),
    }


def render_report(
    kernel_records: list[dict], pipeline_record: dict | None, output: Path
) -> None:
    """Crea e salva un file Markdown con i risultati formattati."""

    def fmt(value: float | None, unit: str = "s") -> str:
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
        "I tempi escludono la compilazione JIT e sono mediane su più ripetizioni.",
        "",
        "| n rilevazioni | Python O(n²) | Numba O(n²) | Numba parallelo "
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
        "- **Numba vs Python**: l'uso del JIT velocizza notevolmente l'esecuzione.",
        "- **Parallelo vs sequenziale**: l'uso di più thread è vantaggioso solo "
        "con moli massicce di dati.",
        "- **O(n log n) vs O(n²)**: essenziale per elaborare dataset estesi.",
        "- **Errore relativo**: minimi scostamenti dovuti all'aritmetica float.",
        "",
    ]

    if pipeline_record:
        lines += [
            "## Pipeline completa sul dataset di produzione",
            "",
            f"- Righe in ingresso: {pipeline_record['input_rows']:,}",
            f"- Righe in uscita: {pipeline_record['output_rows']:,}",
            f"- Cisterne: {pipeline_record['num_tanks']:,}",
            f"- Tempo trasformazioni: {pipeline_record['transform_s']:.3f} s",
            f"- Tempo computazioni HPC: {pipeline_record['hpc_s']:.3f} s",
            f"- Picco memoria Python (`tracemalloc`): "
            f"{pipeline_record['peak_python_memory_mb']:.2f} MB",
            f"- Picco memoria reale del processo: "
            f"{pipeline_record['peak_rss_mb']:.1f} MB",
            f"- Crescita memoria: {pipeline_record['rss_growth_mb']:.1f} MB",
            "",
        ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport scritto in {output}")


def main(argv: list[str] | None = None) -> int:
    """Funzione principale per l'esecuzione dei benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark stress formula.")
    parser.add_argument(
        "--output",
        default="docs/performance_report.md",
        help="File Markdown in cui salvare il report.",
    )
    parser.add_argument(
        "--sensors",
        default="data/full_sensors.tsv",
        help="Percorso dataset sensori.",
    )
    parser.add_argument(
        "--tank-info",
        default="data/full_tank_info.tsv",
        help="Percorso dataset cisterne.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=max(KERNEL_SIZES),
        help="Numero massimo di rilevazioni.",
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
            f"\nDataset {sensors_path} non trovato. Generalo prima con "
            "`python data_generator.py`."
        )

    render_report(kernel_records, pipeline_record, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
