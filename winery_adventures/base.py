"""base comune per tutti i blocchi di analisi della pipeline.

qui definiamo `basewineryanalyzer`, una classe astratta che serve 
come stampo per ogni fase dell'analisi. 

la regola per usarla è una: ogni blocco prende un dataframa polars e 
ne restituisce uno nuovo. questa struttura standard ci permette di collegare 
i passaggi uno dopo l'altro senza errori.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

__all__ = ["BaseWineryAnalyzer"]


class BaseWineryAnalyzer(ABC):
    """interfaccia per ogni singolo passaggio della pipeline.

    ogni classe figlia rappresenta una singola operazione: riceve i dati 
    dal passaggio prims, li elabora e li passa al prossimo.

    è obbligatorio creare il metodo `analyze_data` in ogni classe figlia,
    altrimenti python blocca tutto con un errore.

    example:
        >>> import polars as pl
        >>> class identity(BaseWineryAnalyzer):
        ...     def analyze_data(self, df: pl.DataFrame) -> pl.DataFrame:
        ...         return df
        >>> identity().analyze_data(pl.DataFrame({"a": [1]})).shape
        (1, 1)
    """

    @abstractmethod
    def analyze_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """elabora i dati in ingresso e restituisce il risultato.

        args:
            df: il dataframe polars in arrivo dal passaggio precedente.

        returns:
            un nuovo dataframe polars con i dati elaborati. 
            importante: non modificare i dati originali, polars lavora
            creando oggetti nuovi (è immutabile).

        raises:
            notimplementederror: de una classe figlia non crea questo metodo.
        """
        raise NotImplementedError(
            "le sottoclassi di basewineryanalyzer devono avere analyze_data()."
        )

    def __repr__(self) -> str:  # pragma: no cover - serve solo per il debug
        """testo leggibile per capire quale classe sta lavorando nei log."""
        return f"{self.__class__.__name__}()"