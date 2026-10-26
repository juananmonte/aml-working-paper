"""Run the existing AML functions in order, without executing notebooks."""

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import tomllib
from uuid import uuid4

from loguru import logger

from aml_working_paper.config import PROJ_ROOT, REPORTS_DIR

STAGES = ("data", "preprocess", "train", "routing", "causal")
FEATURES = [
    "economic_context",
    "account_exposure",
    "structuring_score",
    "cross_border_ratio",
    "transaction_volume",
]
STRATEGIES = ("uncertainty", "surprise", "combined", "random")


def load_config(path: Path) -> dict:
    """Read explicit experiment parameters and reject invalid settings."""
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    integer_keys = {
        "seed",
        "n_accounts",
        "periods",
        "window_size",
        "budget",
        "draws",
        "tune",
        "chains",
        "cores",
        "random_repeats",
    }
    expected = integer_keys | {"target_accept"}
    if set(config) != expected:
        raise ValueError(
            f"Config keys: missing={sorted(expected - set(config))}, "
            f"unknown={sorted(set(config) - expected)}"
        )
    for key in integer_keys:
        minimum = 0 if key == "seed" else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    accept = config["target_accept"]
    if type(accept) not in (int, float) or not 0 < accept < 1:
        raise ValueError("target_accept must be between 0 and 1")
    if config["seed"] >= 2**32 - config["random_repeats"]:
        raise ValueError("seed and repeated random seeds must fit in 32 bits")
    if config["n_accounts"] < 10:
        raise ValueError("n_accounts must be >= 10 for the 70/20/10 split")
    if not 3 <= config["window_size"] <= config["periods"]:
        raise ValueError("window_size must be between 3 and periods")
    if config["budget"] * config["window_size"] <= len(FEATURES):
        raise ValueError("Investigation samples must exceed the number of features")
    if config["chains"] < 2:
        raise ValueError("Use at least two chains to inspect convergence")
    return config


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _environment() -> dict:
    """Record software versions and the source checkout when available."""
    result = {"python": platform.python_version(), "platform": platform.platform()}
    result["packages"] = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "pymc",
            "pytensor",
            "arviz",
            "causal-learn",
            "pyarrow",
            "netCDF4",
        )
    }
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        result["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJ_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    lock = PROJ_ROOT / "uv.lock"
    result["lock_sha256"] = (
        hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
    )
    return result


def sampling_diagnostics(trace) -> dict:
    """Extract JSON-safe diagnostics from the locked ArviZ DataTree API."""
    import arviz as az
    import numpy as np

    bfmi = az.bfmi(trace)["energy"].values
    return {
        "divergences": int(trace.sample_stats["diverging"].sum()),
        "bfmi": [float(v) if np.isfinite(v) else None for v in bfmi],
        "note": "Completion does not imply convergence. Inspect R-hat, ESS, divergences and BFMI; demo is a smoke check only.",
    }


def _named_edges(graph, features: list[str]) -> list[str]:
    """Translate exact node names, preserving PC/FCI endpoint symbols."""
    names = {f"X{i + 1}": name for i, name in enumerate(features)}
    edges = []
    for edge in graph.get_graph_edges():
        left, symbol, right = str(edge).split()
        edges.append(f"{names[left]} {symbol} {names[right]}")
    return edges


def _run_causal(val_df, candidates, config: dict, output: Path) -> None:
    import matplotlib
    import pandas as pd

    from aml_working_paper.causal.discovery import discover_fci, discover_pc
    from aml_working_paper.causal.discovery_data import prepare_discovery_data
    from aml_working_paper.causal.evaluation import evaluate_skeleton
    from aml_working_paper.causal.local_sample import prepare_local_windows
    from aml_working_paper.routing.routing_investigate import (
        select_investigations,
        validate_investigations,
    )

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores, graph_edges = [], {}
    for strategy in STRATEGIES:
        selected = select_investigations(
            candidates,
            budget=config["budget"],
            strategy=strategy,
            random_state=config["seed"],
        )
        validate_investigations(selected, candidates, config["budget"])
        if len(selected) != config["budget"]:
            raise ValueError(
                f"Only {len(selected)} windows available for {strategy}; reduce budget"
            )
        selected.to_parquet(
            output / f"data/interim/selected_{strategy}.parquet", index=False
        )
        windows = prepare_local_windows(val_df, selected, FEATURES)
        data = prepare_discovery_data(windows)
        data.to_parquet(
            output / f"data/processed/discovery_{strategy}.parquet", index=False
        )
        graphs = {
            "PC": discover_pc(data, FEATURES).G,
            "FCI": discover_fci(data, FEATURES),
        }
        for method, graph in graphs.items():
            edges = _named_edges(graph, FEATURES)
            graph_edges[f"{method}/{strategy}"] = edges
            scores.append(
                {"method": method, "strategy": strategy, **evaluate_skeleton(edges)}
            )

    evaluation = pd.DataFrame(scores)
    evaluation.to_csv(output / "reports/evaluation.csv", index=False)
    _save_json(output / "reports/edges.json", graph_edges)

    # Repeat random selection as in the explanatory causal notebook (PC only).
    random_scores = []
    for offset in range(config["random_repeats"]):
        seed = config["seed"] + offset
        selected = select_investigations(candidates, config["budget"], "random", seed)
        if len(selected) != config["budget"]:
            raise ValueError(
                f"Random seed {seed} could not meet the investigation budget"
            )
        data = prepare_discovery_data(prepare_local_windows(val_df, selected, FEATURES))
        edges = _named_edges(discover_pc(data, FEATURES).G, FEATURES)
        random_scores.append({"seed": seed, **evaluate_skeleton(edges)})
    pd.DataFrame(random_scores).to_csv(output / "reports/random_pc.csv", index=False)

    ax = (
        evaluation.pivot(index="strategy", columns="method", values="f1")
        .reindex(STRATEGIES)
        .plot.bar()
    )
    ax.set_ylabel("Direct observable skeleton F1 (exploratory)")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "reports/figures/skeleton_f1.png", dpi=150)
    plt.close(ax.figure)


def run_pipeline(config: dict, output_dir: Path, stop_after: str = "causal") -> Path:
    """Execute stages in order and preserve every run in a new directory."""
    import joblib
    import numpy as np

    from aml_working_paper.data.preprocess import preprocess_data
    from aml_working_paper.simulation.equations import simulate_accounts

    if stop_after not in STAGES:
        raise ValueError(f"Unknown stage: {stop_after}")
    output = output_dir.resolve()
    environment = _environment()
    output.mkdir(parents=True, exist_ok=False)
    for relative in (
        "data/raw",
        "data/interim",
        "data/processed",
        "models",
        "reports/figures",
    ):
        (output / relative).mkdir(parents=True)
    _save_json(output / "config.json", config)
    manifest = {
        "status": "running",
        "stop_after": stop_after,
        "completed_stages": [],
        "started_at": datetime.now(UTC).isoformat(),
        "environment": environment,
        "evaluation_scope": "Exploratory direct observable skeleton; pooled temporal windows, not full latent causal recovery",
    }
    started = time.perf_counter()

    def completed(stage: str) -> bool:
        manifest["completed_stages"].append(stage)
        _save_json(output / "run.json", manifest)
        logger.info("Completed {}", stage)
        return stage == stop_after

    try:
        _save_json(output / "run.json", manifest)
        logger.info("Generating synthetic accounts")
        data = simulate_accounts(
            config["n_accounts"], config["periods"], config["seed"]
        )
        data.to_csv(output / "data/raw/synthetic_df_aml.csv", index=False)
        if completed("data"):
            return output

        train, val, test, train_seq, val_seq, _, scaler = preprocess_data(
            data, config["seed"]
        )
        capacity = int(
            (val.groupby("account_id").size() // config["window_size"]).sum()
        )
        if config["budget"] > capacity:
            raise ValueError(
                f"Investigation budget {config['budget']} exceeds validation capacity {capacity}"
            )
        for name, frame in (("train", train), ("val", val), ("test", test)):
            frame.to_parquet(output / f"data/processed/{name}.parquet", index=False)
            np.save(
                output / f"data/interim/{name}_account_ids.npy",
                frame["account_id"].unique(),
            )
        joblib.dump(scaler, output / "models/hmm_scaler.joblib")
        if completed("preprocess"):
            return output

        import arviz as az
        import pymc as pm

        from aml_working_paper.models.bayes_hmm import build_hmm_model
        from aml_working_paper.models.labelswitch import labelswitch

        logger.info(
            "Sampling the HMM: {} draws, {} tune, {} chains",
            config["draws"],
            config["tune"],
            config["chains"],
        )
        model = build_hmm_model(train_seq)
        with model:
            trace = pm.sample(
                draws=config["draws"],
                tune=config["tune"],
                chains=config["chains"],
                cores=config["cores"],
                target_accept=config["target_accept"],
                random_seed=config["seed"],
            )
        aligned, permutations = labelswitch(trace)
        trace.to_netcdf(output / "models/bayesian_hmm_trace_original.nc")
        aligned.to_netcdf(output / "models/bayesian_hmm_aligned_trace.nc")
        np.save(output / "models/label_permutations.npy", permutations)
        variables = ["pi", "current", "mu", "sigma"]
        az.summary(trace, var_names=variables).to_csv(
            output / "reports/hmm_summary_original.csv"
        )
        az.summary(aligned, var_names=variables).to_csv(
            output / "reports/hmm_summary_aligned.csv"
        )
        _save_json(output / "reports/sampling.json", sampling_diagnostics(trace))
        if completed("train"):
            return output

        from aml_working_paper.routing.routing import (
            build_routing_dataframe,
            extract_posterior_draws,
        )
        from aml_working_paper.routing.routing_investigate import (
            build_candidate_windows,
        )

        draws = extract_posterior_draws(aligned)
        routing = build_routing_dataframe(val, val_seq, *draws)
        candidates = build_candidate_windows(routing, config["window_size"])
        routing.to_parquet(output / "data/interim/routing_df.parquet", index=False)
        candidates.to_parquet(output / "data/interim/candidates.parquet", index=False)
        if completed("routing"):
            return output

        _run_causal(val, candidates, config, output)
        completed("causal")
        return output
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if manifest["status"] != "failed":
            manifest["status"] = "completed"
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        _save_json(output / "run.json", manifest)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and start an independently saved experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJ_ROOT / "configs/demo.toml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; an existing path is never overwritten",
    )
    parser.add_argument("--stop-after", choices=STAGES, default="causal")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    label = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    output = args.output_dir or REPORTS_DIR / "runs" / label
    logger.warning(
        "Inspect sampling diagnostics before interpreting results; the demo is an execution check only"
    )
    logger.info("Output directory: {}", output.resolve())
    try:
        result = run_pipeline(config, output, args.stop_after)
    except FileExistsError:
        parser.error(f"Output directory already exists: {output}. Choose a new path.")
    logger.info("Finished. Results: {}", result)


if __name__ == "__main__":
    main()
