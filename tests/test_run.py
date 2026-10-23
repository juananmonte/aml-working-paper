"""Checks for the command-line runner, independent of MCMC convergence."""

import json
from pathlib import Path
import subprocess
import sys

from aml_working_paper.run import load_config, run_pipeline, sampling_diagnostics
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_sampling_diagnostics_supports_locked_arviz_datatree():
    import numpy as np
    import xarray as xr

    stats = xr.Dataset(
        {
            "energy": (
                ("chain", "draw"),
                [[1.0, 2.0, 1.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 2.0, 3.0, 1.0]],
            ),
            "diverging": (("chain", "draw"), np.zeros((2, 6), dtype=bool)),
        }
    )
    trace = xr.DataTree.from_dict({"sample_stats": stats})
    result = sampling_diagnostics(trace)
    assert result["divergences"] == 0
    assert len(result["bfmi"]) == 2
    assert all(np.isfinite(result["bfmi"]))


def test_demo_config_is_explicit():
    config = load_config(ROOT / "configs/demo.toml")
    assert config["seed"] == 42
    assert config["window_size"] <= config["periods"]


@pytest.mark.parametrize(
    "setting",
    ["budget = 0", "n_accounts = 2", "window_size = 999", "draws = true", "typo = 1"],
)
def test_bad_config_is_rejected(tmp_path, setting):
    source = (ROOT / "configs/demo.toml").read_text()
    key = setting.split("=")[0].strip()
    lines = [line for line in source.splitlines() if not line.startswith(key + " =")]
    path = tmp_path / "invalid.toml"
    path.write_text("\n".join(lines) + "\n" + setting + "\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_run_refuses_existing_directory(tmp_path):
    config = load_config(ROOT / "configs/demo.toml")
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        run_pipeline(config, tmp_path, stop_after="data")
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupted_run_is_not_marked_completed(tmp_path, monkeypatch, interruption):
    from aml_working_paper.simulation import equations

    def interrupt(*args):
        raise interruption()

    monkeypatch.setattr(equations, "simulate_accounts", interrupt)
    output = tmp_path / "interrupted"
    with pytest.raises(interruption):
        run_pipeline(load_config(ROOT / "configs/demo.toml"), output, "data")
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["completed_stages"] == []


def test_impossible_budget_is_rejected_before_training(tmp_path):
    config = load_config(ROOT / "configs/demo.toml")
    config["budget"] = 1000
    with pytest.raises(ValueError, match="budget"):
        run_pipeline(config, tmp_path / "too-many-windows", "preprocess")


def test_preprocessing_outputs_are_repeatable_and_accounts_do_not_overlap(tmp_path):
    config = load_config(ROOT / "configs/demo.toml")
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        run_pipeline(config, output, stop_after="preprocess")
    pd.testing.assert_frame_equal(
        pd.read_csv(outputs[0] / "data/raw/synthetic_df_aml.csv"),
        pd.read_csv(outputs[1] / "data/raw/synthetic_df_aml.csv"),
    )
    splits = [
        pd.read_parquet(outputs[0] / f"data/processed/{name}.parquet")
        for name in ("train", "val", "test")
    ]
    accounts = [set(frame["account_id"]) for frame in splits]
    assert (
        sum(len(frame) for frame in splits) == config["n_accounts"] * config["periods"]
    )
    assert accounts[0].isdisjoint(accounts[1])
    assert accounts[0].isdisjoint(accounts[2])
    assert accounts[1].isdisjoint(accounts[2])
    manifest = json.loads((outputs[0] / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["completed_stages"] == ["data", "preprocess"]
    assert (outputs[0] / "models/hmm_scaler.joblib").is_file()
    assert not (outputs[0] / "models/bayesian_hmm_aligned_trace.nc").exists()


def test_help_describes_stage_stopping():
    result = subprocess.run(
        [sys.executable, "-m", "aml_working_paper.run", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--stop-after" in result.stdout
