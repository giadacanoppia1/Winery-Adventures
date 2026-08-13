import polars as pl
from abc import ABC, abstractmethod

class BaseTransformation(ABC):
    """
    Classe base astratta per tutte le trasformazioni dei dati della cantina.
    Garantisce che ogni trasformatore implementi il metodo 'transform'.
    """

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applica una trasformazione ai dati.

        Args:
            df (pl.DataFrame): Il dataframe originale dei sensori.

        Returns:
            pl.DataFrame: Il dataframe trasformato.
        """
        pass

class DataPipeline:
    """Orchestratore: esegue in sequenza una lista di trasformazioni."""

    def __init__(self, transformations: list[BaseTransformation]):
        self.transformations = transformations

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        # Applica il Polars DataFrame a ogni trasformatore uno dopo l'altro
        for transformer in self.transformations:
            df = transformer.transform(df)
        return df