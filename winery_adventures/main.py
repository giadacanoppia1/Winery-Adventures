#!/usr/bin/env python3
r"""Punto di ingresso dell'applicazione Winery Adventures.

Gestisce la pipeline completa (caricamento, trasformazione e calcolo HPC)
sia tramite funzione riutilizzabile sia via riga di comando.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from winery_adventures.computations import WineryHPCComputations
from winery_adventures.logging_config import setup_logging
from winery_adventures.pipeline import WineryPipeline
from winery_adventures.transformations import WineryTransformer

__all__ = ["run_full_pipeline", "main"]

logger = logging.getLogger(__name__)


def run_full_pipeline(
    input_csv: str,
    tank_info_csv: str | None = None,
    output_csv: str | None = None,
    project_name: str | None = None,
    separator: str = "\t",
    n_jobs: int = -1,
) -> pl.DataFrame:
    """Esegue l'intera pipeline dai file di input al file dei risultati."""
    sensors_df = WineryPipeline.load_data(input_csv, separator=separator)

    tank_info_df: pl.DataFrame | None = None
    if tank_info_csv is not None:
        tank_info_df = WineryPipeline.load_data(tank_info_csv, separator=separator)

    pipeline = WineryPipeline(
        [
            WineryTransformer(tank_info_df),
            WineryHPCComputations(n_jobs=n_jobs),
        ],
        project_name=project_name,
    )

    result_df = pipeline.run(sensors_df, log_to_wandb=True)

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_csv(output_path)
        logger.info(
            "Risultati scritti in %s (%d righe).",
            output_path,
            result_df.height,
        )

    return result_df


def _build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser degli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        prog="winery-adventures",
        description="Pipeline di analisi dei sensori.",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="File TSV con le letture dei sensori.",
    )
    parser.add_argument(
        "--tank-info-csv",
        default=None,
        help="File TSV anagrafica cisterne (opzionale).",
    )
    parser.add_argument(
        "--output-csv",
        default="results/results.csv",
        help="File CSV in cui scrivere i risultati.",
    )
    parser.add_argument(
        "--project-name",
        default=None,
        help="Nome del progetto Weights & Biases.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Worker paralleli (-1 = tutti i core).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosità dei log.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Interfaccia a riga di comando principale."""
    args = _build_parser().parse_args(argv)
    setup_logging(args.log_level)

    try:
        run_full_pipeline(
            input_csv=args.input_csv,
            tank_info_csv=args.tank_info_csv,
            output_csv=args.output_csv,
            project_name=args.project_name,
            n_jobs=args.n_jobs,
        )
    except (FileNotFoundError, ValueError, AttributeError):
        logger.exception("Esecuzione della pipeline fallita.")
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
