"""Configurazione di Sphinx per la documentazione di Winery Adventures.

Build::

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html
"""

from __future__ import annotations

import sys
from pathlib import Path

# Il package sta nella cartella superiore rispetto a docs/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]  ))

project = "Winery Adventures"
author = "Gruppo Winery Adventures"
copyright = "2025, Gruppo Winery Adventures"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",  # estrae la documentazione dalle docstring
    "sphinx.ext.napoleon",  # interpreta le docstring in stile Google
    "sphinx.ext.viewcode",  # link al codice sorgente
    "sphinx.ext.intersphinx",  # link alla documentazione delle dipendenze
    "myst_parser",  # permette di includere file Markdown
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "it"

html_theme = "sphinx_rtd_theme"
html_static_path = []

# Le docstring del progetto seguono la convenzione Google.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Numba e wandb non servono per costruire la documentazione: se assenti,
# vengono sostituiti da mock così la build non fallisce.
autodoc_mock_imports = ["numba", "wandb", "joblib"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
