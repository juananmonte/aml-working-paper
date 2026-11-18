# <p align="center"> Bayesian Uncertainty and Surprise for Prioritizing Local Causal Investigations in Anti-Money Laundering (working paper) </p>

##### <p align="center"> [Juan Montenegro](https://github.com/juananmonte), Author<sup>1</sup> Guatemala City, Guatemala; <br>
#
This exploratory study uses a Synthetic Data-Generating Process (DGP) to test whether Bayesian uncertainty and surprise can help decide which account periods to investigate when only a limited number can be reviewed (limited budget). A Bayesian Hidden Markov Model (HMM) estimates the probabilities of hidden AML states. Then, uncertainty and surprise scores are measured in order to prioritize investigation windows of accounts to investigate. After that, local causal discovery is used on those periods to identify potential causal relationships among observed variables. This helps formulating hypothesis about the AML process.

- Bayesian HMM: What latent AML state might an account be in  and how uncertain/surpirsed is the model about a prediction?
- Routing: Given a limited investigation budget, which account windows should be investigated?
- Causal discovery: What potential relationships exist among the observed variables in those selected windows?


In summary, can those discovered relationships help explain the mechanisms associated with latent AML behaviour?


## Project Organization

```
├── LICENSE            <- MIT license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- Description of t
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The datasets for modeling.
│   └── raw            <- The original result of DGP.
│
├── docs               <- Extra information regarding the DGP, HMM, Routing and Causal Inference
|
├── models             <- Trained HMM models
│
├── notebooks          <- Jupyter notebooks for GDP, preprocessing, HMM, Routing and Causal Inference. 
│                         These are alternative and interactive ways to understand how the paper works.
|
├── pyproject.toml     <- Project configuration file with package metadata for
│                         aml_working_paper and configuration for tools
│
├── reports            <- Generated analysis as HTML, PDF, images, etc.
│   ├── figures        <- Generated graphics and figures from the pape
│   └── runs           <- The results from executing the "uv run" command from "reproducing experiments" section
│
└── aml_working_paper   <- Source code (structured pipeline, 5 modules)
    ├── __init__.py             <- Makes aml_working_paper a Python module
    ├── config.py               <- Store useful variables and configuration
    |
    |── simulation              <- Contains the code necessary to preprocess the simulated data      
    │   ├── __init__.py         <- files to make the functions in .py files into modules
    │   └── equations.py        <- Contains the functions necessary to create the observable and hidden variables
    |   └── states.py           <- Functions to calculate the AML state changes
    |
    ├── data                    <- Contains the code necessary to preprocess the simulated data
    |   ├── __init__.py         
    |   └── preprocess.py       <- Contains functions for spliting, transforming and building sequences of data
    |
    ├── models                  <- Model training
    │   ├── __init__.py
    │   └── bayes_hmm.py        <- Build the Hidden Marko Model and it's necessary steps (ex: forward step)
    |   └── labelswitch.py      <- Fix the order of the predicted labels so they match their correct aml state
    |
    ├── routing                 <- Generate the candidate investigation windows
    │   ├── __init__.py
    │   └── routing.py              <- calculate uncertainty and surprise. Generate the routing dataframe.
    |   └── routing_investigate.py  <- Obtain candidate windows and assign a score based on a certain strategy
    |
    └── Causal                  <- Scripts necessary to find the relationships between observable variables
        ├── __init__.py
        └── discovery_data.py   <- Make a dataframe from the selected windows obtained by routing
        └── discovery.py        <- define PC and FCI algorithms
        └── evaluation.py       <- Obtain the different metrics to evaluate PC/FCI results
        └── ground_truth.py     <- Define how the observable and latent variables truly interact with each other
        └── local_sample.py     <- Extract original observations for selected investigation windows.
```


--------

# Research Context

Machine learning models are used in Anti-Money Laundering (AML) systems to identify suspicious transactions and/or unusual account behavior. Nonetheless, these approaches mainly focus on obtaining predictions of possible risk behaviours, risk scores or anomaly scores. These predictions are limited regarding the actual patterns that created an alert. For example, an anomaly detection model such as Isolation Forest can identify unusual observations, but its anomaly score does not establish whether an account is involved in money laundering or explain the relationships among the analyzed variables.

Bayesian models offer an alternative by allowing beliefs to be updated as new evidence becomes available. In particular, Hidden Markov Models (HMMs) can represent changes in latent behavioral states over time. Their predicted probabilities can be used to calculate uncertainty and surprise, providing additional information beyond a single score. Yet, these metrics do not directly establish the causal mechanisms responsible for suspicious behavior. 

Therefore, this study explores whether Bayesian uncertainty and surprise can be used to prioritize account-level time windows for further causal investigation under a limited investigation budget. By combining Bayesian predictions with local causal discovery algorithms, the proposed framework investigates whether uncertainty/surprise-aware prioritization can improve the recovery of causal relationships among observed variables compared to random selection.

# Research Question

## Research Question

**Can Bayesian AML-state uncertainty and surprise help prioritize account-level time windows for local causal investigation, improving causal structure recovery under a limited investigation budget?**

### Key Definitions

- **Investigation budget:** The maximum number of observation windows that can be selected for local causal analysis.
- **Uncertainty:** A measure of how unsure the Bayesian model is about which latent AML state an account occupies.
- **Surprise:** A measure of how unexpected an observation or inferred state is according to the model's estimated probabilities.
- Causal structure recovery: The identification of relationships among observed variables, evaluated against the known reference graph using precision, recall, and F1-score. This does not imply complete causal mechanism recovery or estimation of causal effects.

--------


# Methodology

The experiment consists of five stages:

1. Synthetic data generation. Simulate account trajectories with latent AML states and observed variables, including economic context, network exposure, structuring score, cross-border ratio, and transaction volume. The known data-generating process provides a reference for evaluating recovered relationships.
2. Bayesian state estimation. Fit a Bayesian hidden Markov model to the observed account sequences. Posterior draws are aligned to address label switching before calculating state probabilities and routing scores.
3. Investigation selection. Score candidate observation windows using uncertainty about the latent state, predictive surprise, or a combination of both. Each strategy selects windows under the same investigation budget.
4. Local causal discovery. Apply the PC algorithm to the observations associated with the selected windows to estimate relationships among the observed variables.
5. Evaluation. Compare the recovered graph structure with the reference structure using precision, recall, and F1. Compare the routing strategies with repeated random selection and examine variability across simulation seeds.
The Bayesian model guides where to investigate; the causal discovery stage estimates local graph structure. Uncertainty and surprise serve as selection criteria and are not themselves evidence of a causal relationship.

# Main results

| Strategy    | TP | FP | FN | Precision | Recall | F1-score |
|-------------|----|----|----|-----------|--------|----------|
| Uncertainty | 3  | 2  | 1  | 0.600     | 0.750  | 0.667    |
| **Surprise**| 4  | 2  | 0  | 0.667     | 1.000  | **0.800**|
| Combined    | 3  | 1  | 1  | 0.750     | 0.750  | 0.750    |
| Random      | 3  | 1  | 1  | 0.750     | 0.750  | 0.750    |

Causal skeleton recovery performance across routing strategies under a fixed investigation budget of 20 observation windows. Surprise-based routing achieves the highest F1-score (0.800) and recall (1.000) in this experiment. Results are preliminary and do not establish consistent superiority over random selection.

# Main conclusions so far

- **Bayesian uncertainty and surprise provide alternative criteria for prioritizing local causal investigations.** The proposed framework demonstrates how posterior information from a Bayesian Hidden Markov Model can guide the selection of observation windows under a limited investigation budget, connecting probabilistic state estimation with local causal structure discovery.

- **Surprise-based prioritization shows promising preliminary results for causal structure recovery.** In the reported experiment, surprise-based routing achieved the highest F1-score (0.80) and recall (1.00), outperforming uncertainty-based (0.67 F1), combined (0.75 F1), and random routing (0.75 F1). These results suggest that unexpected observations may contain useful information for causal structure discovery, although further experiments are needed to establish whether this advantage is consistent.

- **Bayesian prioritization does not guarantee better causal structure recovery.** The performance differences among routing strategies indicate that uncertainty and surprise are not necessarily equally informative for causal discovery. The results highlight the importance of evaluating investigation strategies against appropriate baselines rather than assuming that higher Bayesian uncertainty leads to more informative causal investigations.
-------
#
# Limitations

This study is an exploratory pilot with several limitations:

- Synthetic setting. The data follow a specified simulation process and do not capture the full complexity of real financial activity. Results do not establish effectiveness on real AML cases.
- Model assumptions. The Bayesian hidden Markov model provides a simplified representation of account activity. Its uncertainty estimates depend on the model specification, priors, and available observations.Additionally it assumes first-order Markov dependence and conditionally independent Gaussian emission features given the current hidden state. These assumptions may restrict its ability to represent longer-range temporal dependencies and correlations among observed AML indicators.
- Local sample size and selection. Window size and selection affect the observations available for causal discovery. Selecting windows based on model scores may also change the dependence structure within the selected sample.
- Causal discovery assumptions. The PC algorithm relies on assumptions about conditional independence and the underlying causal structure. Latent AML states and temporal dependence require particular care when interpreting relationships among observed variables.
- Scope of evaluation. Graph recovery measures agreement with a specified reference structure. It does not establish that complete causal mechanisms have been recovered or that causal effects have been identified.
- Investigation budget. The budget is represented by the number of selected windows. This simplifies practical investigation costs, which may vary substantially across cases.

--------

# Reproducibility

## Installation

This project uses Python 3.13 and [uv](https://docs.astral.sh/uv/) to manage its environment and dependencies.

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

On   configs/experiment.toml set the desired parameters to evaluate. 

Open a new terminal and run:

```bash
uv run --locked python -m aml_working_paper.run \
  --config configs/experiment.toml \
  --output-dir reports/runs/{experiment-name}
```

The configuration file controls the simulation, Bayesian sampling, window selection, and evaluation settings.

The output directory can be named freely. Its name does not set the experiment seed: the seed is specified in the configuration file.
Use a separate output directory for each experiment to keep results organized.

The `--locked` option requires the dependency lockfile to remain consistent with the project configuration.

### Jupyter Notebooks Alternative

In case there are issues with the previous commands, I have created Jupyter Notebooks to run the entirity of the prjects in a more interactive manner. The order is
1. DGP_interactive
2. Preprocess_interactive
3. Model_interactive
4. Routing_interactive
5. Causal_interactive

--------
These notebooks replicate everything in run.py and have the same guides as the .py files. 


## License

This project is licensed under the [MIT license License](LICENSE).

--------

## Citation

If you use this work, please cite:

```bibtex
@article{author2026project,
  title={Bayesian Uncertainty for Routing Local Causal Investigations},
  author={Juan Montenegro},
  journal={Journal Name},
  volume={},
  pages={},
  year={2026},
  doi={}
}
```

# Acknowledgements

The repository was made using [Cookiecutter Scientific Research](https://github.com/Vncntvx/cookiecutter-scientific-research), which provided the initial project structure and development tooling.
