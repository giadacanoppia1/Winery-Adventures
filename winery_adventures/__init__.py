"""Winery Adventures — pipeline di analisi dei dati di fermentazione.

Il package espone i quattro elementi principali del sistema:

* :class:`~winery_adventures.base.BaseWineryAnalyzer` — contratto astratto
  comune a ogni stadio della pipeline;
* :class:`~winery_adventures.transformations.WineryTransformer` — statistiche
  aggregate sulle letture dei sensori;
* :class:`~winery_adventures.computations.WineryHPCComputations` — formula di
  stress da fermentazione ottimizzata con Numba e joblib;
* :class:`~winery_adventures.pipeline.WineryPipeline` — orchestratore degli
  stadi e logging su Weights & Biases.
"""

from winery_adventures.base import BaseWineryAnalyzer
from winery_adventures.computations import (
    WineryHPCComputations,
    fast_stress_function,
    pairwise_stress_function,
)
from winery_adventures.pipeline import WineryPipeline
from winery_adventures.transformations import WineryTransformer

__version__ = "1.0.0"

__all__ = [
    "BaseWineryAnalyzer",
    "WineryHPCComputations",
    "WineryPipeline",
    "WineryTransformer",
    "fast_stress_function",
    "pairwise_stress_function",
    "__version__",
]

