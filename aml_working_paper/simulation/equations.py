### Contains all the equations necessary for continuous mechanisms
### For the synthetic data, these are used to simulate changes in behaviours of the accounts

import pandas as pd
import numpy as np
from numpy.random import Generator
from aml_working_paper.simulation.states import sample_next_state
from scipy.special import expit


def generate_economic_context(previous_context, rng, rho=0.8, sigma=0.5):
    """
    Generate economic context using an AR(1) process.
    """
    shock = rng.normal(0, sigma)
    economic_context = float(rho * previous_context + shock)
    return economic_context

def generate_account_exposure(economic_context, rng):
    """
    Return a number between 0 an 1 of how much an account is exposed in the network 
    by the amount of interactions one account has with others. 
    Random noise is added so accounts with same economic context don't end up with the same amount of interactions.  
    """
    random_noise = rng.normal(0, 0.7)
    ### -2 and 8 are similar to other paper's synthetic generation data process
    exposure = -2 + 0.8 * economic_context + random_noise
    account_exposure = float(expit(exposure))
    return account_exposure

def generate_aml_score(aml_state, rng):
    """
    Takes an aml_state and gives a score based on the severity of that state. 
    So, a higher aml_state equals a higher aml score.
    Noise is added to make sure higher aml states have a small higher score but not enough to make it too easy for the model
    """
    noise = rng.normal(0, 0.6)
    ### -2 and 1.3 are similar to other paper's synthetic generation data process
    structuring = -2 + 1.3 * aml_state + noise
    aml_score = float(expit(structuring))
    return aml_score

def generate_cross_border_score(account_exposure, aml_score, rng):
    """
    Takes the account exposure and aml score and generates a score regarding the account's interaction
    with other outside networks. 
    This is under the supposition that a higher account exposure and a higher aml score is proportional to
    a higher cross border ratio score. 
    """

    noise = rng.normal(0, 0.6)
    ### -2.5, 1.5 and 0.8 are similar to other paper's synthetic generation data process
    linear_predictor = (-2.5 + 1.5 * account_exposure + 0.8 * aml_score + noise)
    cross_border_ratio_score = float(expit(linear_predictor))
    return cross_border_ratio_score

def generate_transaction_volume(economic_context, aml_state, structuring_score, rng):
    """
    It generates the volume of transactions for an account
    """

    log_mean = (2.5 + 0.25 * economic_context + 0.20 * aml_state + 0.60 * structuring_score)
    volume = rng.lognormal(log_mean, 0.3)
    return float(volume)

def simulate_account(periods=100, initial_aml_state=0, seed=42, economic_context_path=None, initial_economic_context=0.0):
    """
    
    """
    
    seed_sequence = np.random.SeedSequence(seed)

    if economic_context_path is None:
        child_seeds = seed_sequence.spawn(6)

        rng_economic = np.random.default_rng(child_seeds[0])
        start = 1

    else:
        child_seeds = seed_sequence.spawn(
        rng_network = np.random.default_rng(child_seeds[start]),
        rng_state = np.random.default_rng(child_seeds[start + 1]),
        rng_structuring = np.random.default_rng(child_seeds[start + 2]),
        rng_cross_border = np.random.default_rng(child_seeds[start + 3]),
        rng_volume = np.random.default_rng(child_seeds[start + 4])*5)
        start = 0

    # Check the number of periods
    if periods <= 0:
        raise ValueError("Periods must be positive")

    # Check the initial AML state
    if initial_aml_state not in [0, 1, 2]:
        raise ValueError("Invalid AML state")

    # Check the economic context path
    if economic_context_path is not None:
        if len(economic_context_path) != periods:
            raise ValueError("Invalid economic context length")
    
    seed_sequence = np.random.SeedSequence(seed)

    if economic_context_path is None:
        child_seeds = seed_sequence.spawn(6)

        rng_economic = np.random.default_rng(child_seeds[0])
        start = 1

    else:
        child_seeds = seed_sequence.spawn(5)
        start = 0

    rng_network = np.random.default_rng(child_seeds[start])
    rng_state = np.random.default_rng(child_seeds[start + 1])
    rng_structuring = np.random.default_rng(child_seeds[start + 2])
    rng_cross_border = np.random.default_rng(child_seeds[start + 3])
    rng_volume = np.random.default_rng(child_seeds[start + 4])

    economic_context = initial_economic_context
    previous_aml_state = initial_aml_state

    rows = []

    for t in range(periods):

        # 1. Economic context
        if economic_context_path is None:
            economic_context = generate_economic_context(economic_context, rng_economic)
        else:
            economic_context = float(economic_context_path[t])
        # 2. Network exposure
        network_exposure = generate_account_exposure(economic_context, rng_network)
        # 3. AML state
        aml_state = sample_next_state(previous_state=previous_aml_state, network_exposure=network_exposure, rng=rng_state)
        # 4. Structuring score
        aml_score = generate_aml_score(aml_state, rng_structuring)
        # 5. Cross-border ratio
        cross_border_score = generate_cross_border_score(network_exposure, aml_state, rng_cross_border)
        # 6. Transaction volume
        transaction_volume = generate_transaction_volume(economic_context, aml_state, aml_score, rng_volume)

        # Save observation
        rows.append({
            "time": t,
            "economic_context": economic_context,
            "network_exposure": network_exposure,
            "previous_aml_state": previous_aml_state,
            "aml_state": aml_state,
            "structuring_score": aml_score,
            "cross_border_ratio": cross_border_score,
            "transaction_volume": transaction_volume
        })

        # Update previous state
        previous_aml_state = aml_state

    return pd.DataFrame(rows)

def simulate_economic_context(periods, seed=42, initial_context=0.0):
    rng = np.random.default_rng(seed)
    context = initial_context
    values = []
    for _ in range(periods):
        context = generate_economic_context(context, rng)
        values.append(context)

    return np.array(values)

def simulate_accounts(n_accounts=100, periods=100, seed=42):
    # Create seeds for the economy and each account
    seeds = np.random.SeedSequence(seed).spawn(n_accounts + 1)
    # Generate the shared economic context
    economic_seed = int(seeds[0].generate_state(1)[0])
    economic_path = simulate_economic_context(periods, economic_seed)
    # Simulate each account
    accounts = []

    for account_id in range(n_accounts):
        account_seed = int(seeds[account_id + 1].generate_state(1)[0])
        account = simulate_account(periods=periods, seed=account_seed, economic_context_path=economic_path)
        account.insert(0, "account_id", account_id)
        accounts.append(account)

    # Combine all accounts into one dataset
    return pd.concat(accounts, ignore_index=True)

