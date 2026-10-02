# AGENTS.md — aml-working-paper (`structured` / `python`)

Shared guidance for all coding agents working in this repository.


## Quick Start — Python

```bash
uv venv --python 3.13 && source .venv/bin/activate
uv sync
make lint          # ruff format --check && ruff check
make format        # ruff format && ruff check --fix
```

Package management: always `uv add` / `uv sync` / `uv run` — never `pip`.



## Programming Philosophy (Structured — pipeline-first)

**Default: layered pipeline in `aml_working_paper/` / `src/`.**

- `data/make_dataset.py` / `make_dataset.jl` — raw → processed
- `features/build_features.py` — feature engineering (`def main`)
- `models/train_model.py` — train & persist (`def main`)
- `visualization/visualize.py` — figures to `reports/figures`
- `utils/tools.py` / `utils.jl` — shared helpers only

Each module exposes `def main(input_path, output_path)` with type hints and `if __name__ == "__main__":`.

| Task | Location |
|------|----------|
| One-off exploration | `notebooks/` or `scripts/` |
| Reusable pipeline | `aml_working_paper/` |
| Repeated 3× logic | promote to `utils/` |




## Data & Storage Convention

```
data/
├── raw/        # immutable originals (any format)
├── interim/    # intermediate + cross-language (*.arrow)
├── processed/  # final analysis (ALWAYS Parquet)
└── external/   # third-party
reports/figures/  logs/  models/
```

- Raw may be any format; processed **always Parquet** (Python) .
- By size: `<1GB pandas` / `<10GB Polars` / `>10GB DuckDB` (Python).

## Code Style Guidelines

### Python
- Line length 88, double quotes, import order stdlib → third-party → local
- Type hints Python 3.12 style: `list[int]`, `int | None`
- Docstring sklearn-style, raise `ValueError` with message, `loguru` + `tqdm`
- Lint/format: `ruff` (`make lint` / `make format`)



## Project Architecture

```
aml_working_paper/
├── config.py
├── data/make_dataset.py
├── features/build_features.py
├── models/train_model.py
├── visualization/visualize.py
└── utils/tools.py
scripts/.gitkeep  # use for one-offs
```


### Configuration
- Python: `aml_working_paper/config.py` (dotenv + loguru + tqdm)
- Julia: `src/config.jl` (mirrors Python paths)

## Skills
- uv package manager: `.agents/skills/uv-package-manager/SKILL.md`

