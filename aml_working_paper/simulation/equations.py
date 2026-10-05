### Contains all the equations necessary for continuous mechanisms
### For the synthetic data, these are used to simulate changes in behaviours of the accounts

import pandas as pd
import numpy as np
from numpy.random import Generator
from aml_working_paper.simulation.states import sample_next_state


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

def generate_economic_context(
    previous_context: float,
    rng: Generator,
    rho: float = 0.8,
    sigma: float = 0.5,
) -> float:
    """Generate economic context using an AR(1) process."""

    shock = rng.normal(loc=0.0, scale=sigma)

    economic_context = (
        rho * previous_context
        + shock
    )

    return float(economic_context)


def simulate_account(
    periods: int = 100,
    initial_aml_state: int = 0,
    seed: int = 42,
    economic_context_path: np.ndarray | None = None,
    initial_economic_context: float = 0.0,
) -> pd.DataFrame:
    """Simulate one account over multiple time periods."""

    if periods <= 0:
        raise ValueError("periods must be greater than 0.")

    if initial_aml_state not in {0, 1, 2}:
        raise ValueError(
            "initial_aml_state must be one of {0, 1, 2}."
        )

    if (
        economic_context_path is not None
        and len(economic_context_path) != periods
    ):
        raise ValueError(
            "economic_context_path must have length equal to periods."
        )

    seed_sequence = np.random.SeedSequence(seed)

    if economic_context_path is None:
        child_seeds = seed_sequence.spawn(6)

        rng_economic = np.random.default_rng(child_seeds[0])
        rng_network = np.random.default_rng(child_seeds[1])
        rng_state = np.random.default_rng(child_seeds[2])
        rng_structuring = np.random.default_rng(child_seeds[3])
        rng_cross_border = np.random.default_rng(child_seeds[4])
        rng_volume = np.random.default_rng(child_seeds[5])

    else:
        child_seeds = seed_sequence.spawn(5)

        rng_network = np.random.default_rng(child_seeds[0])
        rng_state = np.random.default_rng(child_seeds[1])
        rng_structuring = np.random.default_rng(child_seeds[2])
        rng_cross_border = np.random.default_rng(child_seeds[3])
        rng_volume = np.random.default_rng(child_seeds[4])

    economic_context = initial_economic_context
    previous_aml_state = initial_aml_state

    rows = []

    for t in range(periods):

        if economic_context_path is None:
            economic_context = generate_economic_context(
                previous_context=economic_context,
                rng=rng_economic,
            )
        else:
            economic_context = float(
                economic_context_path[t]
            )

        network_exposure = generate_network_exposure(
            economic_context=economic_context,
            rng=rng_network,
        )

        aml_state = sample_next_state(
            previous_state=previous_aml_state,
            network_exposure=network_exposure,
            rng=rng_state,
        )

        structuring_score = generate_structuring_score(
            aml_state=aml_state,
            rng=rng_structuring,
        )

        cross_border_ratio = generate_cross_border_ratio(
            network_exposure=network_exposure,
            aml_state=aml_state,
            rng=rng_cross_border,
        )

        transaction_volume = generate_transaction_volume(
            economic_context=economic_context,
            aml_state=aml_state,
            structuring_score=structuring_score,
            rng=rng_volume,
        )

        rows.append(
            {
                "time": t,
                "economic_context": economic_context,
                "network_exposure": network_exposure,
                "previous_aml_state": previous_aml_state,
                "aml_state": aml_state,
                "structuring_score": structuring_score,
                "cross_border_ratio": cross_border_ratio,
                "transaction_volume": transaction_volume,
            }
        )

        previous_aml_state = aml_state

    return pd.DataFrame(rows)

def simulate_economic_context(
    periods: int,
    seed: int = 42,
    initial_context: float = 0.0,
) -> np.ndarray:
    """Simulate a shared economic context path."""

    if periods <= 0:
        raise ValueError(
            "periods must be greater than 0."
        )

    rng = np.random.default_rng(seed)

    context = initial_context
    values = []

    for _ in range(periods):
        context = generate_economic_context(
            previous_context=context,
            rng=rng,
        )

        values.append(context)

    return np.asarray(values)

def simulate_accounts(
    n_accounts: int = 100,
    periods: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate multiple accounts over a shared economic context."""

    if n_accounts <= 0:
        raise ValueError(
            "n_accounts must be greater than 0."
        )

    if periods <= 0:
        raise ValueError(
            "periods must be greater than 0."
        )

    seed_sequence = np.random.SeedSequence(seed)

    child_seeds = seed_sequence.spawn(
        n_accounts + 1
    )

    economic_seed = int(
        child_seeds[0].generate_state(1)[0]
    )

    economic_context_path = simulate_economic_context(
        periods=periods,
        seed=economic_seed,
    )

    account_frames = []

    for account_id in range(n_accounts):

        account_seed = int(
            child_seeds[
                account_id + 1
            ].generate_state(1)[0]
        )

        account_df = simulate_account(
            periods=periods,
            seed=account_seed,
            economic_context_path=economic_context_path,
        )

        account_df.insert(
            0,
            "account_id",
            account_id,
        )

        account_frames.append(
            account_df
        )

    return pd.concat(
        account_frames,
        ignore_index=True,
    )

