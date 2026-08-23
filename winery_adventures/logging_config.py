"""configurazione generale dei messaggi di sistema (log).

ogni parte del programma genera i suoi messaggi, ma questo file decide
come e dove mostrarli a schermo, impostando una regola unica per tutti.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["setup_logging"]

#: aspetto del messaggio: data e ora, importanza, chi lo ha mandato e testo.
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

#: come scrivere la data e l'ora.
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int | str = logging.INFO) -> None:
    """prepara il sistema per stampare i messaggi nel terminale.

    args:
        level: l'importanza minima dei messaggi da mostrare (ad esempio
            "info" per le cose normali, "debug" per tutti i dettagli).

    note:
        usando force=true, se chiami questa funzione due volte per sbaglio 
        non si creano doppioni nei messaggi, ma la configurazione si aggiorna.
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )