# Report di performance — formula di stress da fermentazione

Report generato da `benchmarks/benchmark_stress.py`.

## Ambiente di esecuzione

- Sistema: Windows-10-10.0.26200-SP0
- Processore: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel
- Python: 3.11.9
- NumPy: 2.4.6 · Polars: 1.43.2

## Confronto tra implementazioni

I tempi escludono la compilazione JIT e sono mediane su più ripetizioni.

| n rilevazioni | Python O(n²) | Numba O(n²) | Numba parallelo | Numba O(n log n) | Speed-up vs Python | Errore relativo |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 8.32 ms | 12.7 µs | 47.1 µs | 4.5 µs | 1,849× | 7.61e-16 |
| 500 | 159.82 ms | 215.4 µs | 142.3 µs | 31.0 µs | 5,155× | 2.02e-15 |
| 1,000 | 503.52 ms | 807.5 µs | 263.9 µs | 92.5 µs | 5,443× | 1e-14 |
| 5,000 | — | 27.97 ms | 5.34 ms | 683.6 µs | — | 2.26e-15 |
| 20,000 | — | 333.20 ms | 85.93 ms | 3.10 ms | — | 7.94e-15 |
| 100,000 | — | — | — | 15.66 ms | — | — |

### Lettura dei risultati

- **Numba vs Python**: l'uso del JIT velocizza notevolmente l'esecuzione.
- **Parallelo vs sequenziale**: l'uso di più thread è vantaggioso solo con moli massicce di dati.
- **O(n log n) vs O(n²)**: essenziale per elaborare dataset estesi.
- **Errore relativo**: minimi scostamenti dovuti all'aritmetica float.

## Pipeline completa sul dataset di produzione

- Righe in ingresso: 100,000
- Righe in uscita: 300,000
- Cisterne: 100
- Tempo trasformazioni: 0.039 s
- Tempo computazioni HPC: 0.042 s
- Picco memoria Python (`tracemalloc`): 0.90 MB
- Picco memoria reale del processo: 0.0 MB
- Crescita memoria: 0.0 MB
