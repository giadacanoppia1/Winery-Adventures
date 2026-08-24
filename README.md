# Winery Adventures

Pipeline di elaborazione e analisi dei dati provenienti dai sensori delle
cisterne di fermentazione di una piccola cantina. Il sistema legge le letture
grezze di pH, temperatura e volume di mosto, le arricchisce con statistiche
aggregate e calcola per ogni cisterna un **indice di stress da fermentazione**
che segnala il rischio di fermentazione irregolare o bloccata.

> La consegna originale del progetto è archiviata in
> [`docs/project_brief.md`](docs/project_brief.md).

---

## Indice

- [Installazione](#installazione)
- [Uso](#uso)
- [Architettura](#architettura)
- [L'indice di stress da fermentazione](#lindice-di-stress-da-fermentazione)
- [Performance](#performance)
- [Sviluppo](#sviluppo)
- [Struttura del repository](#struttura-del-repository)

---

## Installazione

Serve **Python 3.10 o superiore**.

```bash
git clone <url-del-repository>
cd Winery-Adventures

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Installazione in modalità sviluppo, con test e strumenti di qualità
pip install -e ".[dev]"
```

Per la sola esecuzione, senza strumenti di sviluppo:

```bash
pip install -r requirements.txt
```

Verifica dell'installazione:

```bash
pytest
```

### Weights & Biases

Il logging delle metriche richiede un account W&B. Al primo utilizzo:

```bash
wandb login
```

Per lavorare senza account (o in CI) basta disattivare la sincronizzazione:

```bash
export WANDB_MODE=offline
```

La pipeline non si interrompe se W&B non è disponibile: l'errore viene
registrato nei log e l'analisi prosegue.

---

## Uso

### Da riga di comando

```bash
python -m winery_adventures.main \
    --input-csv data/sensors_sample.tsv \
    --tank-info-csv data/tank_info_sample.tsv \
    --output-csv results/results.csv
```

Opzioni disponibili:

| Opzione | Default | Descrizione |
| --- | --- | --- |
| `--input-csv` | *obbligatoria* | File TSV con le letture dei sensori |
| `--tank-info-csv` | `None` | Anagrafica delle cisterne; se omessa, l'analisi per vitigno viene saltata |
| `--output-csv` | `results/results.csv` | File CSV dei risultati |
| `--project-name` | `winery-adventures` | Progetto Weights & Biases |
| `--n-jobs` | `-1` | Worker paralleli (`-1` = tutti i core) |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Come libreria

```python
from winery_adventures import (
    WineryHPCComputations,
    WineryPipeline,
    WineryTransformer,
)

sensors = WineryPipeline.load_data("data/sensors_sample.tsv")
tank_info = WineryPipeline.load_data("data/tank_info_sample.tsv")

pipeline = WineryPipeline(
    [WineryTransformer(tank_info), WineryHPCComputations()],
    project_name="cantina-2025",
)
result = pipeline.run(sensors, log_to_wandb=True)

# Le cinque cisterne più a rischio
print(
    result.select("tank_id", "stress_score")
    .unique(subset=["tank_id"])
    .sort("stress_score", descending=True)
    .head(5)
)
```

### Dati di input

**`sensors_*.tsv`** — una riga per lettura:

| Colonna | Tipo | Descrizione |
| --- | --- | --- |
| `tank_id` | intero | Identificativo della cisterna |
| `time` | stringa | Istante della lettura |
| `pH` | decimale | Acidità del mosto |
| `temp` | decimale | Temperatura in °C |
| `quantity_liters` | intero | Volume di mosto; può essere mancante |

**`tank_info_*.tsv`** — una riga per cisterna:

| Colonna | Tipo | Descrizione |
| --- | --- | --- |
| `tank_id` | intero | Identificativo della cisterna |
| `grape_variety` | stringa | Blend di vitigni separati da virgola |
| `capacity_liters` | intero | Capienza della cisterna |

### Colonne prodotte

| Colonna | Significato |
| --- | --- |
| `avg_pH_per_tank` | pH medio di tutte le letture della cisterna |
| `tank_num_readings` | Numero di letture della cisterna |
| `grape_variety_num_readings` | Letture complessive che riguardano quel vitigno |
| `temperature_deviation` | Scostamento assoluto da 26 °C |
| `temperature_deviation_scaled` | Scostamento normalizzato su 1000 litri |
| `stress_score` | Indice di stress da fermentazione della cisterna |

Se l'anagrafica delle cisterne è disponibile, l'output è **esploso per
vitigno**: una lettura di una cisterna con tre vitigni genera tre righe.

### Dataset di grandi dimensioni

```bash
python data_generator.py     # genera data/full_sensors.tsv (100 000 righe)
```

I file generati non sono versionati: si ricreano con questo comando.

---

## Architettura

```
TSV ──▶ Data Loading ──▶ Trasformazioni ──▶ Computazioni HPC ──▶ CSV + W&B
```

Ogni stadio implementa lo stesso contratto astratto,
`BaseWineryAnalyzer.analyze_data(df) -> df`, e la pipeline si limita ad
applicarli in sequenza. Aggiungere una fase non richiede di modificare
l'orchestratore.

| Modulo | Responsabilità |
| --- | --- |
| `base.py` | Contratto astratto comune a ogni stadio |
| `transformations.py` | Statistiche aggregate sulle letture |
| `computations.py` | Formula di stress ottimizzata (Numba + joblib) |
| `pipeline.py` | Orchestrazione degli stadi e logging su W&B |
| `main.py` | Composizione degli stadi e interfaccia a riga di comando |
| `logging_config.py` | Configurazione centralizzata del logging |

I diagrammi UML (classi, sequenza, casi d'uso) e le motivazioni delle scelte
di progetto sono in [`docs/architecture.md`](docs/architecture.md); i sorgenti
PlantUML in [`docs/uml/`](docs/uml/).

---

## L'indice di stress da fermentazione

Lo stress di una cisterna è la media, su tutte le coppie di rilevazioni, della
loro divergenza pesata sul volume:

$$
\text{stress} = \frac{1}{n^2}\sum_{i=0}^{n-1}\sum_{j=0}^{n-1}
\Bigl(|pH_i - pH_j| + 2\,|T_i - T_j|\Bigr)
\left(\frac{500}{q_i} + \frac{500}{q_j}\right)
$$

Forte variabilità di pH e temperatura indica una fermentazione poco
controllata; il fattore di volume amplifica il segnale nelle cisterne piccole,
termicamente meno stabili.

Il modulo `computations.py` ne offre tre implementazioni equivalenti:

| Funzione | Complessità | Ruolo |
| --- | --- | --- |
| `pairwise_stress_python` | O(n²) | Traduzione letterale della specifica; oracolo di correttezza |
| `pairwise_stress_function` | O(n²) | Stessa formula compilata JIT con Numba |
| `pairwise_stress_function_parallel` | O(n²) | Ciclo esterno distribuito con `prange` |
| `fast_stress_function` | **O(n log n)** | Riscrittura algebrica, usata in produzione |

La versione veloce sfrutta la simmetria della formula. Posti `w_i = 500/q_i` e
`D_ij = |pH_i − pH_j| + 2|T_i − T_j|`:

$$
\sum_{i,j} D_{ij}(w_i + w_j) = 2\sum_i w_i \sum_j D_{ij}
$$

e la somma interna delle distanze assolute si calcola per tutti gli indici in
un'unica passata su dati ordinati, con somme prefisse. I test verificano che le
quattro implementazioni coincidano entro la precisione della macchina
(errore relativo ~10⁻¹⁵).

---

## Performance

Misure su 100 000 letture e 100 cisterne — il report completo, riproducibile,
è in [`docs/performance_report.md`](docs/performance_report.md):

| n rilevazioni | Python O(n²) | Numba O(n²) | Numba O(n log n) |
| ---: | ---: | ---: | ---: |
| 1 000 | 767 ms | 1,44 ms | 126 µs |
| 20 000 | — | 585 ms | 4,2 ms |
| 100 000 | — | — | 27 ms |

Pipeline completa sul dataset di produzione: **0,08 s** di trasformazioni e
**0,11 s** di computazioni HPC, con una crescita della memoria residente di
circa 35 MB.

Per rigenerare il report sulla propria macchina:

```bash
python data_generator.py
python benchmarks/benchmark_stress.py --output docs/performance_report.md
```

Le ottimizzazioni adottate:

- **Compilazione JIT** dei kernel numerici con Numba (`cache=True` evita di
  ricompilare a ogni avvio);
- **riscrittura algoritmica** da O(n²) a O(n log n);
- **parallelizzazione per cisterna** con joblib su thread — possibile perché i
  kernel rilasciano il GIL (`nogil=True`), il che evita di serializzare gli
  array verso processi separati;
- **window functions di Polars** (`.over()`) al posto di `group_by` + `join`:
  un solo passaggio sui dati;
- **proiezione delle colonne** prima del partizionamento, per non copiare dati
  che il kernel non usa.

---

## Sviluppo

### Test

```bash
pytest                                  # tutti i test
pytest -m "not slow"                    # esclude i test di accettazione
pytest --cov --cov-report=term-missing  # con copertura
```

I test in `tests/unit/` e `tests/acceptance/` sono la **specifica fornita dal
committente** e non vanno modificati. I test aggiuntivi scritti dal gruppo sono
nei file con suffisso `_extended` e coprono casi limite, gestione degli errori
ed equivalenza tra le implementazioni della formula.

> **Nota sulla copertura.** `coverage.py` non può tracciare le righe compilate
> in codice macchina da Numba: misurata normalmente, `computations.py`
> risulterebbe scoperto anche con tutti i test verdi. Il job `coverage` della
> CI esegue quindi la suite con `NUMBA_DISABLE_JIT=1`, che fa girare i kernel
> come normale codice Python (97% di copertura complessiva).

### Qualità del codice

```bash
ruff format .    # formattazione automatica
ruff check .     # lint: PEP 8, import, naming, docstring, bug latenti
```

La configurazione è in `pyproject.toml`. Per eseguire i controlli
automaticamente a ogni commit:

```bash
pip install pre-commit
pre-commit install
```

### Documentazione

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Le docstring seguono la convenzione Google e vengono estratte da
`sphinx.ext.autodoc` con `napoleon`.

### Integrazione continua

Il workflow `.github/workflows/ci.yml` esegue a ogni push e pull request:

1. **lint** — `ruff format --check` e `ruff check`;
2. **test** — suite completa su Python 3.10, 3.11 e 3.12;
3. **coverage** — copertura con JIT disattivato;
4. **smoke** — esecuzione end-to-end sui dati di esempio.

### Convenzioni di collaborazione

- Un branch per funzionalità (`nome`), mai commit diretti su `main`.
- Ogni modifica entra tramite pull request, con revisione e CI verde.
- Le attività sono tracciate come issue e collegate alla PR che le chiude.

---

## Struttura del repository

```
Winery-Adventures/
├── winery_adventures/          # package principale
│   ├── base.py                 # contratto astratto della pipeline
│   ├── transformations.py      # statistiche aggregate
│   ├── computations.py         # formula di stress (Numba + joblib)
│   ├── pipeline.py             # orchestrazione e logging W&B
│   ├── main.py                 # composizione degli stadi e CLI
│   └── logging_config.py       # configurazione del logging
├── tests/
│   ├── unit/                   # test unitari (forniti + aggiuntivi)
│   ├── acceptance/             # test di accettazione end-to-end
│   └── conftest.py             # fixture condivise
├── benchmarks/
│   └── benchmark_stress.py     # profilazione e generazione del report
├── docs/
│   ├── architecture.md         # scelte di progetto e diagrammi Mermaid
│   ├── performance_report.md   # report di performance (generato)
│   ├── project_brief.md        # consegna originale
│   ├── uml/                    # sorgenti PlantUML dei diagrammi
│   ├── conf.py                 # configurazione Sphinx
│   └── index.rst, api.rst      # struttura della documentazione
├── data/                       # dataset di esempio
├── data_generator.py           # generatore del dataset di grandi dimensioni
├── pyproject.toml              # dipendenze, build, ruff, pytest, coverage
└── .github/workflows/ci.yml    # integrazione continua
```

---

## Licenza

Distribuito con licenza MIT — vedi [`LICENSE`](LICENSE).
