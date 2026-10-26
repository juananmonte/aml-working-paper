# Running the AML experiment

The command-line runner calls the existing Python functions. Notebooks remain explanatory and are not executed by the runner. Their reading order will be documented in the README.

## Environment

Install Git and uv, clone this repository, then run from its root:

```bash
uv sync --locked --python 3.13
```

The lockfile fixes package versions. Python 3.13, the compiler/numerical libraries and hardware can still influence compilation time and floating-point/MCMC results. A fixed seed does not promise byte-identical posterior samples on different machines.

## Commands

```bash
# Short full execution: checks the wiring, not posterior convergence.
uv run --locked python -m aml_working_paper.run

# Current notebook settings: 100 accounts, 100 periods,
# 500 tuning iterations and 500 draws, four chains, budget 10.
uv run --locked python -m aml_working_paper.run --config configs/experiment.toml

# Stop after preprocessing, without fitting the HMM.
uv run --locked python -m aml_working_paper.run --stop-after preprocess

# Save to a particular NEW directory.
uv run --locked python -m aml_working_paper.run --output-dir reports/runs/my-demo

# Available options.
uv run --locked python -m aml_working_paper.run --help
```

`--stop-after` accepts `data`, `preprocess`, `train`, `routing`, or `causal` (default). Each command starts from simulation; it does not resume a previous run or reuse a saved trace. Copy a TOML file to customize parameters. Unknown, missing or invalid parameters are rejected.

If Make is available, `make demo`, `make experiment`, `make reproduce CONFIG=configs/experiment.toml`, `make train`, and `make routing` provide shortcuts. Python commands also work without Make, including on Windows.

## Outputs

Each run creates a new directory under `reports/runs/`, or the path passed to `--output-dir`. Existing paths are refused. Tracked datasets, models and notebook outputs are preserved. Generated runs are ignored by Git.

| Relative path inside a run | Content |
|---|---|
| `config.json` | Exact input parameters |
| `run.json` | Status, completed stages, elapsed time, Python/package versions, Git commit/dirty state, lockfile hash |
| `data/raw/synthetic_df_aml.csv` | Regenerated synthetic panel |
| `data/processed/{train,val,test}.parquet` | Account splits, with transformed HMM emissions |
| `data/interim/*_account_ids.npy` | Account identifiers in each split |
| `models/hmm_scaler.joblib` | Scaler fitted on training accounts only |
| `models/*trace*.nc` | Original and aligned HMM posterior samples |
| `models/label_permutations.npy` | Alignment permutations |
| `reports/hmm_summary_*.csv`, `reports/sampling.json` | Posterior diagnostics, divergences, BFMI |
| `data/interim/routing_df.parquet` | Routing scores for all validation accounts |
| `data/interim/candidates.parquet` | Candidate investigation windows |
| `data/interim/selected_*.parquet` | Selected windows for each strategy |
| `data/processed/discovery_*.parquet` | Observable investigation data, retaining window/account IDs |
| `reports/evaluation.csv`, `reports/edges.json` | PC/FCI results for uncertainty, surprise, combined and random routing |
| `reports/random_pc.csv` | PC evaluation under repeated random selections |
| `reports/figures/skeleton_f1.png` | Strategy comparison using this run's scores |

Test accounts are saved but are not used for model fitting, routing or discovery. The current exploratory comparison uses validation accounts, matching the notebooks.

## Interpreting completion

`status: completed` means execution reached the requested stage. Inspect R-hat, ESS, divergences and BFMI before interpreting the HMM. The demo intentionally uses very short chains. The experiment configuration copies current notebook settings and is not a guarantee of adequate convergence.

Evaluation compares undirected **direct observable connections** with the existing ground truth. PC/FCI currently pool temporally dependent investigation windows, and the synthetic system contains a hidden AML state. These scores are exploratory comparisons; they do not establish recovery of the full latent causal mechanism or validate the assumptions of the independence tests. Multiple random repetitions measure variation from window selection, not uncertainty across independently generated experiments.

On failure, the runner exits with an error and retains available files with `status: failed` in `run.json`. Start again using a new output directory after addressing the error.

## Runner checks

```bash
uv run --locked python -m pytest tests/test_run.py
```

These checks cover configuration validation, no-overwrite behavior, repeatable data generation, disjoint account splits, stage stopping, and command-line help. They do not test posterior convergence or prove scientific claims.
