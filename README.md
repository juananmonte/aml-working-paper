# <p align="center"> Bayesian Uncertainty for Routing Local Causal Investigations: A Synthetic AML working paper </p>

##### <p align="center"> [Juan Montenegro](https://github.com/juananmonte), Author<sup>1</sup>

##### <p align="center"> <sup>1</sup>Guatemala City, Guatemala; <br>

#
The following working paper is an exploratory study of whether bayesian uncertainty regading anti-money laundering (AML) states can help select apropiate observation windows. The information in these windows are then passed through local causal investigation networks under a limited investigation budget.

This working paper combines synthetic account trajectories, a Bayesian Hidden Markov Model (HMM), a window selection process based on uncertainty and predictive surprise, and local causal discovery. Selection strategies are compared with random selection by measuring recovery of known relationships in the data-generating process.

This draft examines the approach in a controlled synthetic setting. Its findings are preliminary and do not establish effectiveness on real AML data.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter/Marimo notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         aml_working_paper and configuration for tools
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── logs           <- Experiment logs
│
├── tests              <- Test directory
│
└── aml_working_paper   <- Source code (structured pipeline, 5 modules)
    ├── __init__.py             <- Makes aml_working_paper a Python module
    ├── config.py               <- Store useful variables and configuration
    ├── data                    <- Data acquisition and loading
    │   ├── __init__.py
    │   └── make_dataset.py     <- Scripts to download or generate data (raw → processed Parquet)
    ├── features                <- Feature engineering
    │   ├── __init__.py
    │   └── build_features.py   <- Code to create features for modeling
    ├── models                  <- Model training
    │   ├── __init__.py
    │   └── train_model.py      <- Code to train and persist models
    ├── visualization           <- Data visualization
    │   ├── __init__.py
    │   └── visualize.py        <- Code to create visualizations
    └── utils                   <- Utility functions
        ├── __init__.py
        └── tools.py            <- General utilities
```


--------

# Research Context

Anti-money laundering monitoring involves deciding which account activity deserves closer investigation. When investigation capacity is limited, selecting cases becomes part of the analytical problem.

A model may assign similar risk estimates to observations while expressing different levels of uncertainty about their underlying states. This raises a question: could that uncertainty help identify cases where further investigation would be informative?

This study explores that possibility in a synthetic setting. A Bayesian hidden Markov model estimates latent AML states, and uncertainty and predictive surprise guide the selection of observation windows for local causal discovery. Because the data-generating process is known, the recovered relationships can be compared with a predefined reference structure.

The objective is to assess whether these selection strategies improve structural recovery relative to random selection under the same investigation budget

#
# Research Question

Can Bayesian AML-state uncertainty identify which cases deserve local causal investigation, improving causal mechanism recovery under a limited investigation budget?

For this working paper , the investigation budget is  the number of observation windows selected for analysis. Causal recovery is assessed through recovery of relationships in the reference graph; it does not imply recovery of complete causal mechanisms or estimation of causal effects.

--------

## Installation

### System Requirements

- ​**OS**: Linux/macOS/Windows (Tested on Ubuntu 22.04 LTS)
- ​**Python**: ≥3.10
- ​**Hardware**: [Specify if needed, e.g., NVIDIA GPU with CUDA 12.x]

### Environment Setup

```bash
# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate  # Unix/macOS
# .venv\Scripts\activate   # Windows
uv sync
```

### Synthetic Data Generation

1. Download datasets from [DOI: 10.xxxx/zenodo.xxxxxx]/[Google Drive link]
2. Place raw data in `data/raw/` following our [data documentation](docs/data_standards.md)
3. Run preprocessing:

```bash
make data
```

--------

# Methodology

The experiment consists of five stages:

1. Synthetic data generation. Simulate account trajectories with latent AML states and observed variables, including economic context, network exposure, structuring score, cross-border ratio, and transaction volume. The known data-generating process provides a reference for evaluating recovered relationships.
2. Bayesian state estimation. Fit a Bayesian hidden Markov model to the observed account sequences. Posterior draws are aligned to address label switching before calculating state probabilities and routing scores.
3. Investigation selection. Score candidate observation windows using uncertainty about the latent state, predictive surprise, or a combination of both. Each strategy selects windows under the same investigation budget.
4. Local causal discovery. Apply the PC algorithm to the observations associated with the selected windows to estimate relationships among the observed variables.
5. Evaluation. Compare the recovered graph structure with the reference structure using precision, recall, and F1. Compare the routing strategies with repeated random selection and examine variability across simulation seeds.
The Bayesian model guides where to investigate; the causal discovery stage estimates local graph structure. Uncertainty and surprise serve as selection criteria and are not themselves evidence of a causal relationship.

### Experimental Design

| Component          | Implementation Details          |
|--------------------|----------------------------------|
| Data Processing    | [Brief description]             |  
| Model Architecture | [Key technical specifications]   |
| Training Protocol  | [Optimization strategy]          |
| Evaluation Metrics | [List metrics with justification]|

# Limitations

This study is an exploratory pilot with several limitations:

- Synthetic setting. The data follow a specified simulation process and do not capture the full complexity of real financial activity. Results do not establish effectiveness on real AML cases.
- Model assumptions. The Bayesian hidden Markov model provides a simplified representation of account activity. Its uncertainty estimates depend on the model specification, priors, and available observations.
- Local sample size and selection. Window size and selection affect the observations available for causal discovery. Selecting windows based on model scores may also change the dependence structure within the selected sample.
- Causal discovery assumptions. The PC algorithm relies on assumptions about conditional independence and the underlying causal structure. Latent AML states and temporal dependence require particular care when interpreting relationships among observed variables.
- Scope of evaluation. Graph recovery measures agreement with a specified reference structure. It does not establish that complete causal mechanisms have been recovered or that causal effects have been identified.
- Investigation budget. The budget is represented by the number of selected windows. This simplifies practical investigation costs, which may vary substantially across cases.

--------

## Reproducibility

## Installation

This project uses Python 3.13 and [uv](https://docs.astral.sh/uv/) to manage its environment and dependencies.

Clone the repository:

```bash
git clone https://github.com/juananmonte/aml-working-paper.git
cd aml-working-paper
```

Install the dependencies using the committed lockfile:

```bash
uv sync --locked
```

## Reproducing the experiments

Run the complete pipeline from the repository root:

```bash
uv run --locked python -m aml_working_paper.run \
  --config configs/experiment.toml \
  --output-dir reports/runs/research-seed42
```

The configuration file controls the simulation, Bayesian sampling, window selection, and evaluation settings.

The output directory can be named freely. Its name does not set the experiment seed: the seed is specified in the configuration file.
Use a separate output directory for each experiment to keep results organized.

The `--locked` option requires the dependency lockfile to remain consistent with the project configuration.


--------

## License

This project is licensed under the [MIT license License](LICENSE).

--------

## Citation

If you use this work, please cite:

```bibtex
@article{author2025project,
  title={Title of Your Paper},
  author={Author1 and Author2},
  journal={Journal Name},
  volume={X},
  pages={xx--xx},
  year={2025},
  doi={10.xxxx/xxxxx}
}
```

# Acknowledgements

The repository was made using [Cookiecutter Scientific Research](https://github.com/Vncntvx/cookiecutter-scientific-research), which provided the initial project structure and development tooling.
