"""gestore principale dell'elaborazione dati della cantina.

questa classe prende una lista di analizzatori e li esegue uno dopo l'altro.
il risultato del primo diventa i dati di partenza del secondo, e così via.
se in futuro vorrai aggiungere un nuovo filtro, ti basterà inserirlo nella
lista senza dover toccare questo codice.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import wandb

from winery_adventures.base import BaseWineryAnalyzer

__all__ = ["WineryPipeline"]

logger = logging.getLogger(__name__)


class WineryPipeline:
    """esegue i vari passaggi dell'analisi e salva i risultati su w&b.

    Args:
        analyzers: la lista dei passaggi da eseguire sui dati.
        project_name: nome del progetto weights & biases per le metriche.

    Example:
        >>> from winery_adventures.transformations import WineryTransformer
        >>> pipeline = WineryPipeline([WineryTransformer()])
        >>> isinstance(pipeline.analyzers, list)
        True
    """

    #: progetto w&b di default se non ne specifichi uno.
    DEFAULT_PROJECT_NAME: str = "winery-adventures"

    #: la colonna finale che ci interessa salvare su w&b.
    METRIC_COLUMN: str = "stress_score"

    def __init__(
        self,
        analyzers: Sequence[BaseWineryAnalyzer],
        project_name: str | None = None,
    ) -> None:
        """Salva la lista delle operazioni da fare e il progetto w&b."""
        self.analyzers = list(analyzers)
        self.project_name = project_name or self.DEFAULT_PROJECT_NAME

    # ------------------------------------------------------------------
    # caricamento dati
    # ------------------------------------------------------------------
    @staticmethod
    def load_data(path: str | Path, separator: str = "\t") -> pl.DataFrame:
        """Legge un file di testo e lo trasforma in un dataframe polars.

        Args:
            path: dove si trova il file.
            separator: il carattere che divide le colonne.

        Returns:
            i dati caricati e pronti all'uso.

        Raises:
            filenotfounderror: se il file non esiste.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"file di input non trovato: {file_path}")

        df = pl.read_csv(file_path, separator=separator)
        logger.info("caricato %s: %d righe, %d colonne.", file_path, df.height, df.width)
        return df

    # ------------------------------------------------------------------
    # esecuzione
    # ------------------------------------------------------------------
    def run(self, df: pl.DataFrame, log_to_wandb: bool = False) -> pl.DataFrame:
        """Esegue tutti i passaggi dell'analisi, uno per uno.

        Args:
            df: i dati grezzi iniziali (es. i sensori).
            log_to_wandb: se vero, alla fine manda le metriche a w&b.

        Returns:
            i dati finali elaborati.
        """
        for analyzer in self.analyzers:
            logger.info("esecuzione stadio %r.", analyzer)
            df = analyzer.analyze_data(df)

        if log_to_wandb:
            self.log_to_wandb(df)

        return df

    def log_to_wandb(self, df: pl.DataFrame) -> None:
        """Invia a weights & biases lo stress calcolato per ogni cisterna.

        inviamo una sola riga per cisterna (non una per ogni singola lettura),
        così non intasiamo i grafici. se la connessione a w&b fallisce,
        il programma ignora l'errore e va avanti, perché l'analisi è più
        importante del salvataggio del grafico.

        Args:
            df: i dati finali che contengono la colonna dello stress.
        """
        if self.METRIC_COLUMN not in df.columns:
            logger.warning(
                "colonna '%s' assente: nessuna metrica da inviare a w&b.",
                self.METRIC_COLUMN,
            )
            return

        try:
            wandb.init(project=self.project_name)

            columns = [self.METRIC_COLUMN]
            if "tank_id" in df.columns:
                columns.insert(0, "tank_id")
                # prendiamo solo un valore per cisterna per non duplicare i dati
                metrics = df.select(columns).unique(
                    subset=["tank_id"], maintain_order=True
                )
            else:
                metrics = df.select(columns)

            for row in metrics.iter_rows(named=True):
                wandb.log(row)

            # calcoliamo e inviamo le statistiche generali
            wandb.log(
                {
                    f"{self.METRIC_COLUMN}_mean": metrics[self.METRIC_COLUMN].mean(),
                    f"{self.METRIC_COLUMN}_max": metrics[self.METRIC_COLUMN].max(),
                    "num_tanks": metrics.height,
                }
            )
            logger.info("inviate a w&b le metriche di %d cisterne.", metrics.height)
        except Exception:  # noqa: BLE001 - se w&b fallisce, non bloccare il programma
            logger.exception("logging su weights & biases non riuscito.")
        finally:
            try:
                wandb.finish()
            except Exception:  # noqa: BLE001 - chiudiamo il processo al meglio che possiamo
                logger.debug("chiusura della run w&b non riuscita.", exc_info=True)
