"""Trasformazioni sui dati dei sensori: pH medio, conteggi, deviazione termica.

Usa window function di Polars (.over()): un solo passaggio sul DataFrame,
senza group_by + join.
"""

from __future__ import annotations

import logging

import polars as pl

from winery_adventures.base import BaseWineryAnalyzer

__all__ = ["WineryTransformer"]

logger = logging.getLogger(__name__)


class WineryTransformer(BaseWineryAnalyzer):
    """Arricchisce le letture dei sensori con statistiche aggregate per cisterna."""

    STANDARD_TEMPERATURE: float = 26.0
    REFERENCE_VOLUME_LITERS: float = 1000.0

    def __init__(self, tank_info_df: pl.DataFrame | None = None) -> None:
        """Salva l'anagrafica delle cisterne (opzionale)."""
        self.tank_info_df = tank_info_df

    def add_avg_ph_per_tank(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggiunge ``avg_pH_per_tank``: pH medio di tutte le letture della cisterna."""
        self._require_columns(df, ["tank_id", "pH"], "add_avg_ph_per_tank")
        return df.with_columns(
            pl.col("pH").mean().over("tank_id").alias("avg_pH_per_tank")
        )

    def add_num_readings_per_tank(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggiunge ``tank_num_readings``: numero di letture per cisterna."""
        self._require_columns(df, ["tank_id"], "add_num_readings_per_tank")
        return df.with_columns(
            pl.len().over("tank_id").alias("tank_num_readings"),
        )

    def add_num_readings_per_grape_variety(self, df: pl.DataFrame) -> pl.DataFrame:
        """Esplode le letture per vitigno e conta le rilevazioni per vitigno.

        Una cisterna ospita più vitigni: dopo il join con l'anagrafica, una
        lettura di una cisterna con tre vitigni genera tre righe.
        """
        if self.tank_info_df is None:
            raise AttributeError(
                "Nessuna anagrafica delle cisterne disponibile: "
                "WineryTransformer è stato costruito con tank_info_df=None, "
                "impossibile calcolare le rilevazioni per vitigno."
            )
        self._require_columns(df, ["tank_id"], "add_num_readings_per_grape_variety")

        tank_info = self._normalize_tank_info(self.tank_info_df)

        return (
            df.join(tank_info, on="tank_id", how="left")
            .explode("grape_variety")  # una riga per (lettura, vitigno)
            .with_columns(
                pl.len().over("grape_variety").alias("grape_variety_num_readings")
            )
        )

    def add_temperature_deviation(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggiunge la deviazione di temperatura dal valore standard (26°C).

        Se disponibile il volume, aggiunge anche la versione normalizzata su
        1000 litri: la stessa deviazione pesa di più in una cisterna piccola.
        """
        self._require_columns(df, ["temp"], "add_temperature_deviation")

        deviation = (pl.col("temp") - self.STANDARD_TEMPERATURE).abs()
        out = df.with_columns(deviation.alias("temperature_deviation"))

        if "quantity_liters" not in df.columns:
            logger.debug(
                "Colonna 'quantity_liters' assente: deviazione non normalizzata."
            )
            return out

        scaled = deviation * (self.REFERENCE_VOLUME_LITERS / pl.col("quantity_liters"))
        return out.with_columns(scaled.alias("temperature_deviation_scaled"))

    def analyze_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Applica in sequenza tutte le trasformazioni disponibili.

        Le statistiche per cisterna vanno calcolate prima dell'esplosione per
        vitigno, altrimenti tank_num_readings risulterebbe gonfiato.
        """
        logger.info("Trasformazioni su %d righe in ingresso.", df.height)

        df = self.add_avg_ph_per_tank(df)
        df = self.add_num_readings_per_tank(df)

        if self.tank_info_df is not None:
            df = self.add_num_readings_per_grape_variety(df)
        else:
            logger.warning("Anagrafica cisterne assente: salto l'analisi per vitigno.")

        df = self.add_temperature_deviation(df)

        logger.info("Trasformazioni completate: %d righe in uscita.", df.height)
        return df

    @staticmethod
    def _normalize_tank_info(tank_info: pl.DataFrame) -> pl.DataFrame:
        """Converte ``grape_variety`` in lista, se è ancora una stringa CSV."""
        if "grape_variety" not in tank_info.columns:
            raise ValueError(
                "L'anagrafica delle cisterne deve contenere la colonna 'grape_variety'."
            )
        if tank_info.schema["grape_variety"] == pl.String:
            return tank_info.with_columns(pl.col("grape_variety").str.split(","))
        return tank_info

    @staticmethod
    def _require_columns(df: pl.DataFrame, columns: list[str], caller: str) -> None:
        """Solleva ValueError se mancano colonne richieste da una trasformazione."""
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(
                f"{caller}: colonne mancanti nel DataFrame: {sorted(missing)}. "
                f"Colonne disponibili: {df.columns}"
            )
