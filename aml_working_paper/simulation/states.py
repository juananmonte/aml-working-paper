### Satesp.py is setting a discreet latent state. It's the sanity check. 
### It's use is to prove that our suppositions work before continuing with equations.py

import numpy as np
from numpy.random import Generator
from scipy.special import softmax
import pytest

BASE_TRANSITION = np.array(
    [
        [0.94, 0.05, 0.01],
        [0.10, 0.75, 0.15],
        [0.03, 0.17, 0.80],
    ],
    dtype=float,
)


def validate_transition_matrix(matrix: np.ndarray) -> None:
    """Validate a Markov transition matrix."""

    if matrix.ndim != 2:
        raise ValueError("Transition matrix must be two-dimensional.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Transition matrix must be square.")

    if np.any(matrix < 0.0):
        raise ValueError("Transition probabilities cannot be negative.")

    row_sums = matrix.sum(axis=1)

    if not np.allclose(row_sums, 1.0):
        raise ValueError("Each row of the transition matrix must sum to 1.")


def sample_next_state(
    previous_state: int,
    network_exposure: float,
    rng: Generator,
) -> int:
    """Sample the next latent AML state."""

    probabilities = transition_probabilities(
        previous_state=previous_state,
        network_exposure=network_exposure,
    )

    next_state = rng.choice(
        a=np.arange(len(probabilities)),
        p=probabilities,
    )

    return int(next_state)

def test_sample_next_state_returns_valid_state():
    rng = np.random.default_rng(42)

    state = sample_next_state(
        previous_state=0,
        network_exposure=0.5,
        rng=rng,
    )

    assert state in {0, 1, 2}

def test_transition_probabilities_sum_to_one():
    probabilities = transition_probabilities(
        previous_state=1,
        network_exposure=0.5,
    )

    assert np.isclose(probabilities.sum(), 1.0)

def test_higher_exposure_increases_active_aml_probability():
    low = transition_probabilities(
        previous_state=0,
        network_exposure=0.0,
    )

    high = transition_probabilities(
        previous_state=0,
        network_exposure=1.0,
    )

    assert high[2] > low[2]

def transition_probabilities(previous_state: int, network_exposure_effect, network_exposure: float, transition_matrix: np.ndarray = BASE_TRANSITION,
    ) -> np.ndarray:
    """Compute transition probabilities conditional on network exposure."""

    if not 0.0 <= network_exposure <= 1.0:
        raise ValueError(
        "network_exposure must be between 0 and 1."
    )

    base_probabilities = transition_matrix[previous_state]

    base_logits = np.log(base_probabilities)

    exposure_shift = (
        network_exposure
        * network_exposure_effect
    )

    adjusted_logits = base_logits + exposure_shift

    softmax_values = softmax(adjusted_logits)

    return softmax_values

def test_invalid_network_exposure_raises_error():
    with pytest.raises(ValueError):
        transition_probabilities(
            previous_state=0,
            network_exposure=1.5,
        )   