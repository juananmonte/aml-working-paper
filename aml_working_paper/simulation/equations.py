### Contains all the equations necessary for continuous mechanisms
### For the synthetic data, these are used to simulate changes in behaviours of the accounts


import numpy as np
from numpy.random import Generator


def logistic(x: float | np.ndarray) -> float | np.ndarray:
    """Map real-valued inputs into the interval (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))

def generate_network_exposure(economic_context: float,rng: Generator,) -> float:
    """Generate network exposure from economic context.
    Haré
    economic_context:
    rng: generate random sample BUT with random seed in order to replicate results
    """

    noise = rng.normal(loc=0.0, scale=0.7)

    linear_predictor = (
        -2.0
        + 0.8 * economic_context
        + noise
    )

    return float(logistic(linear_predictor))

def generate_structuring_score(
    aml_state: int,
    rng: Generator,
) -> float:
    """Generate structuring behaviour from latent AML state."""

    noise = rng.normal(loc=0.0, scale=0.6)

    linear_predictor = (
        -2.0
        + 1.3 * aml_state
        + noise
    )

    return float(logistic(linear_predictor))

def generate_cross_border_ratio(
    network_exposure: float,
    aml_state: int,
    rng: Generator,
) -> float:
    """Generate cross-border activity."""

    noise = rng.normal(loc=0.0, scale=0.6)

    linear_predictor = (
        -2.5
        + 1.5 * network_exposure
        + 0.8 * aml_state
        + noise
    )

    return float(logistic(linear_predictor))

def generate_transaction_volume(
    economic_context: float,
    aml_state: int,
    structuring_score: float,
    rng: Generator,
) -> float:
    """Generate positive transaction volume."""

    log_mean = (
        2.5
        + 0.25 * economic_context
        + 0.20 * aml_state
        + 0.60 * structuring_score
    )

    return float(
        rng.lognormal(
            mean=log_mean,
            sigma=0.3,
        )
    )