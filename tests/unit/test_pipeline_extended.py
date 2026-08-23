"""Test aggiuntivi su trasformazioni, orchestratore e punto di ingresso.

Coprono la gestione degli errori, i formati alternativi dell'anagrafica delle
cisterne e l'esecuzione della pipeline dalla riga di comando.
"""

import polars as pl
import pytest

from winery_adventures.main import main, run_full_pipeline
from winery_adventures.pipeline import WineryPipeline
from winery_adventures.transformations import WineryTransformer


# ----------------------------------------------------------------------
# Trasformazioni
# ----------------------------------------------------------------------
def test_missing_columns_raise_value_error():
    """Ogni trasformazione segnala esplicitamente le colonne mancanti."""
    transformer = WineryTransformer()

    with pytest.raises(ValueError, match="colonne mancanti"):
        transformer.add_avg_ph_per_tank(pl.DataFrame({"tank_id": [1]}))

    with pytest.raises(ValueError, match="colonne mancanti"):
        transformer.add_temperature_deviation(pl.DataFrame({"tank_id": [1]}))


def test_tank_info_accepts_unsplit_string(sensors_df, tank_info_df):
    """L'anagrafica funziona sia con i vitigni in stringa sia già splittati.

    Nei file TSV il blend è una stringa separata da virgole; nei test viene
    passato come lista. Entrambi i formati devono dare lo stesso risultato.
    """
    from_string = WineryTransformer(tank_info_df).analyze_data(sensors_df)
    from_list = WineryTransformer(
        tank_info_df.with_columns(pl.col("grape_variety").str.split(","))
    ).analyze_data(sensors_df)

    assert from_string.height == from_list.height == 9
    assert from_string["grape_variety"].to_list() == from_list["grape_variety"].to_list()


def test_analyze_data_without_tank_info_skips_grape_stage(sensors_df):
    """Senza anagrafica l'analisi per vitigno viene saltata, non fallisce."""
    out = WineryTransformer().analyze_data(sensors_df)

    assert "avg_pH_per_tank" in out.columns
    assert "grape_variety_num_readings" not in out.columns
    assert out.height == sensors_df.height


def test_transformations_do_not_mutate_input(sensors_df):
    """Gli stadi restituiscono nuovi DataFrame senza alterare l'input."""
    columns_before = list(sensors_df.columns)
    WineryTransformer().analyze_data(sensors_df)

    assert sensors_df.columns == columns_before


def test_tank_num_readings_is_not_inflated_by_explosion(
    sensors_df, tank_info_df_grape_variety_split
):
    """Il conteggio per cisterna resta quello reale anche dopo l'esplosione.

    Se le statistiche per cisterna fossero calcolate dopo il join con
    l'anagrafica, ogni lettura verrebbe contata una volta per vitigno.
    """
    out = WineryTransformer(tank_info_df_grape_variety_split).analyze_data(sensors_df)

    assert out.filter(pl.col("tank_id") == 1)["tank_num_readings"][0] == 2
    assert out.filter(pl.col("tank_id") == 2)["tank_num_readings"][0] == 1


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def test_load_data_missing_file_raises():
    """Un percorso inesistente produce un errore esplicito."""
    with pytest.raises(FileNotFoundError):
        WineryPipeline.load_data("percorso/inesistente.tsv")


def test_load_data_reads_sample_dataset():
    """Il dataset di esempio viene letto con il separatore corretto."""
    df = WineryPipeline.load_data("data/sensors_sample.tsv")

    assert {"tank_id", "pH", "temp", "quantity_liters"} <= set(df.columns)
    assert df.height == 100


def test_empty_pipeline_returns_input(sensors_df):
    """Una pipeline senza stadi è l'identità."""
    assert WineryPipeline([]).run(sensors_df) is sensors_df


def test_log_to_wandb_without_metric_column_is_noop(monkey_wandb_run, sensors_df):
    """Se manca la colonna delle metriche non si apre nemmeno una run W&B."""
    WineryPipeline([]).log_to_wandb(sensors_df)

    assert monkey_wandb_run.logs == []


def test_log_to_wandb_logs_one_entry_per_tank(monkey_wandb_run):
    """Si logga una riga per cisterna, non una per lettura."""
    df = pl.DataFrame(
        {
            "tank_id": [1, 1, 1, 2],
            "stress_score": [0.5, 0.5, 0.5, 0.9],
        }
    )

    WineryPipeline([], project_name="Test").log_to_wandb(df)

    per_tank = [log for log in monkey_wandb_run.logs if "tank_id" in log]
    assert len(per_tank) == 2


# ----------------------------------------------------------------------
# Punto di ingresso
# ----------------------------------------------------------------------
def test_run_full_pipeline_without_tank_info(
    tmp_path, monkey_wandb_run, monkey_joblib, sensors_df
):
    """La pipeline funziona anche senza anagrafica delle cisterne."""
    sensors_path = tmp_path / "sensors.tsv"
    sensors_path.write_text(sensors_df.write_csv(separator="\t"))
    output_path = tmp_path / "out.csv"

    result = run_full_pipeline(
        input_csv=str(sensors_path),
        tank_info_csv=None,
        output_csv=str(output_path),
        project_name="Test",
    )

    assert output_path.is_file()
    assert result.height == sensors_df.height
    assert "stress_score" in result.columns
    assert "grape_variety_num_readings" not in result.columns


def test_cli_returns_error_code_on_missing_file(tmp_path):
    """La CLI non solleva eccezioni all'utente: restituisce un codice d'uscita."""
    exit_code = main(
        [
            "--input-csv",
            str(tmp_path / "inesistente.tsv"),
            "--output-csv",
            str(tmp_path / "out.csv"),
        ]
    )

    assert exit_code == 1


def test_cli_runs_on_sample_data(tmp_path, monkey_wandb_run, monkey_joblib):
    """Esecuzione completa dalla riga di comando sui dati di esempio."""
    output_path = tmp_path / "results.csv"

    exit_code = main(
        [
            "--input-csv",
            "data/sensors_sample.tsv",
            "--tank-info-csv",
            "data/tank_info_sample.tsv",
            "--output-csv",
            str(output_path),
            "--n-jobs",
            "1",
        ]
    )

    assert exit_code == 0
    result = pl.read_csv(output_path)
    assert {"avg_pH_per_tank", "stress_score"} <= set(result.columns)
