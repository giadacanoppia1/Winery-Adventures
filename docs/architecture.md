# Architettura del sistema

Questo documento descrive la struttura di Winery Adventures e le scelte di
progetto che la sostengono. I diagrammi sono in Mermaid, così da essere
renderizzati direttamente da GitHub; le versioni PlantUML, più dettagliate,
sono in [`docs/uml/`](uml/) e si generano con `plantuml docs/uml/*.puml`.

## Visione d'insieme

Il sistema è una pipeline a stadi. Ogni stadio riceve un `polars.DataFrame`,
lo arricchisce e lo passa al successivo. L'orchestratore non conosce i tipi
concreti degli stadi: dipende solo dal contratto astratto
`BaseWineryAnalyzer`.

```mermaid
flowchart LR
    A[("TSV sensori<br/>+ anagrafica")] --> B[Data Loading<br/><i>WineryPipeline.load_data</i>]
    B --> C[Data Transformation<br/><i>WineryTransformer</i>]
    C --> D[Data Analysis<br/><i>WineryHPCComputations</i>]
    D --> E[Reporting<br/><i>CSV + Weights &amp; Biases</i>]
```

## Diagramma delle classi

```mermaid
classDiagram
    class BaseWineryAnalyzer {
        <<abstract>>
        +analyze_data(df) DataFrame*
        +__repr__() str
    }

    class WineryTransformer {
        +STANDARD_TEMPERATURE: float = 26.0
        +REFERENCE_VOLUME_LITERS: float = 1000.0
        -tank_info_df: DataFrame|None
        +analyze_data(df) DataFrame
        +add_avg_ph_per_tank(df) DataFrame
        +add_num_readings_per_tank(df) DataFrame
        +add_num_readings_per_grape_variety(df) DataFrame
        +add_temperature_deviation(df) DataFrame
        -_normalize_tank_info(tank_info) DataFrame
        -_require_columns(df, columns, caller) void
    }

    class WineryHPCComputations {
        +REQUIRED_COLUMNS: tuple
        +OUTPUT_COLUMN: str = "stress_score"
        -n_jobs: int
        -use_fast_algorithm: bool
        -pairwise_threshold: int
        +analyze_data(df) DataFrame
        -_execute(tasks) list~float~
    }

    class WineryPipeline {
        +DEFAULT_PROJECT_NAME: str
        +METRIC_COLUMN: str = "stress_score"
        -analyzers: list~BaseWineryAnalyzer~
        -project_name: str
        +load_data(path, separator) DataFrame
        +run(df, log_to_wandb) DataFrame
        +log_to_wandb(df) void
    }

    class StressKernels {
        <<module>>
        +pairwise_stress_python(pH, temp, quantity) float
        +pairwise_stress_function(pH, temp, quantity) float
        +pairwise_stress_function_parallel(pH, temp, quantity) float
        +fast_stress_function(pH, temp, quantity) float
        -_sum_abs_differences(values) ndarray
        -_compute_tank_stress(...) float
    }

    class Main {
        <<module>>
        +run_full_pipeline(input_csv, tank_info_csv, output_csv) DataFrame
        +main(argv) int
    }

    BaseWineryAnalyzer <|-- WineryTransformer
    BaseWineryAnalyzer <|-- WineryHPCComputations
    WineryPipeline o-- BaseWineryAnalyzer : analyzers
    WineryHPCComputations ..> StressKernels : invoca
    Main ..> WineryPipeline : compone
    Main ..> WineryTransformer : istanzia
    Main ..> WineryHPCComputations : istanzia
```

## Diagramma di sequenza

```mermaid
sequenceDiagram
    actor User as Enologo
    participant Main as run_full_pipeline
    participant Pipe as WineryPipeline
    participant Tr as WineryTransformer
    participant HPC as WineryHPCComputations
    participant JL as joblib.Parallel
    participant K as kernel Numba
    participant WB as wandb

    User->>Main: run_full_pipeline(input, tank_info, output)
    Main->>Pipe: load_data(TSV)
    Pipe-->>Main: sensors_df, tank_info_df
    Main->>Pipe: run(df, log_to_wandb=True)

    Pipe->>Tr: analyze_data(df)
    Tr->>Tr: pH medio e conteggi per cisterna
    Tr->>Tr: join + explode per vitigno
    Tr->>Tr: deviazione di temperatura
    Tr-->>Pipe: DataFrame arricchito

    Pipe->>HPC: analyze_data(df)
    HPC->>HPC: filtra righe non valide
    HPC->>HPC: partition_by(tank_id)
    HPC->>JL: Parallel(prefer="threads")(tasks)
    loop per ogni cisterna
        JL->>K: stress(pH, temp, quantity)
        K-->>JL: stress_score
    end
    JL-->>HPC: punteggi
    HPC-->>Pipe: DataFrame con stress_score

    Pipe->>WB: init / log / finish
    Pipe-->>Main: DataFrame finale
    Main-->>User: results.csv
```

## Diagramma dei casi d'uso

```mermaid
flowchart TB
    subgraph Attori
        E([Enologo])
        D([Data Engineer])
    end

    subgraph Sistema["Winery Adventures"]
        UC1(["Analizzare lo stato<br/>di fermentazione"])
        UC2(["Individuare le<br/>cisterne a rischio"])
        UC3(["Consultare le<br/>metriche su W&amp;B"])
        UC4(["Eseguire la pipeline<br/>da riga di comando"])
        UC5(["Generare il report<br/>di performance"])
        UC6(["Generare un<br/>dataset di prova"])
        UC7(["Calcolare lo stress<br/>da fermentazione"])
        UC8(["Contare le rilevazioni<br/>per vitigno"])
    end

    E --> UC1
    E --> UC2
    E --> UC3
    D --> UC4
    D --> UC5
    D --> UC6
    UC1 -. include .-> UC7
    UC1 -. extend .-> UC8
    UC2 -. include .-> UC7
    UC4 -. include .-> UC1
```

## Scelte di progetto

### Perché una classe astratta e non semplici funzioni

Ogni stadio della pipeline espone lo stesso metodo `analyze_data(df) -> df`.
Questo permette a `WineryPipeline.run()` di essere un ciclo di tre righe, del
tutto indipendente dagli stadi concreti: si possono aggiungere fasi (per
esempio un rilevatore di anomalie) senza toccare l'orchestratore. È il
principio aperto/chiuso applicato al caso più semplice possibile.

### Perché la deviazione di temperatura è normalizzata sul volume

Una deviazione di 2 °C in una cisterna da 500 litri è ben più preoccupante
della stessa deviazione in una da 2000 litri: la massa termica maggiore rende
la fermentazione più stabile. La normalizzazione su 1000 litri rende
confrontabili cisterne di taglia diversa.

### Perché due livelli di parallelismo

Il calcolo dello stress è indipendente cisterna per cisterna: è quindi
parallelizzabile "in orizzontale" con joblib. Poiché i kernel Numba rilasciano
il GIL (`nogil=True`), il backend a **thread** è preferibile a quello a
processi: niente serializzazione degli array, niente copie di memoria. Per il
caso limite di una singola cisterna con milioni di rilevazioni resta
disponibile `pairwise_stress_function_parallel`, che parallelizza il ciclo
interno con `prange`.

### Perché tre implementazioni della stessa formula

`pairwise_stress_python` è la traduzione letterale della specifica e funge da
oracolo di correttezza. `pairwise_stress_function` è la stessa formula
compilata JIT: dimostra il guadagno dovuto al solo Numba, a parità di
algoritmo. `fast_stress_function` cambia l'algoritmo e porta il costo da
$O(n^2)$ a $O(n \log n)$. Tenerle tutte e tre permette di misurare separatamente il
contributo della compilazione e quello della riprogettazione algoritmica, ed è
esattamente ciò che il report di performance mostra.

Il dettaglio della fattorizzazione algebrica è documentato nella docstring di
`fast_stress_function`; l'idea è che, essendo $D_{ij}$ simmetrico,

$$\sum_{i,j} D_{ij} (w_i + w_j) = 2 \sum_i w_i \sum_j D_{ij}$$

e la somma interna $\sum_j |x_i - x_j|$ si calcola per tutti gli $i$ in un'unica
passata su dati ordinati, tramite somme prefisse.
### Gestione dei dati mancanti

Il generatore di dati produce circa il 10% di letture con `quantity_liters`
nullo. Le righe con valori mancanti vengono escluse dal calcolo dello stress
ma **restano** nel DataFrame di output: manca il dato, non la lettura.
Sostituire i null con una media inventerebbe informazione e falserebbe una
formula che misura proprio la variabilità.
