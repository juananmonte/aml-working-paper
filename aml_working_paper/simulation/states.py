import numpy as np
import pandas as pd
from scipy.special import softmax


### The values in base transitions are darbitrary. Nonetheless, the values have been selected in such a way that
### 1. There is consistence between periods: diagonal values are higher in order to state that there are changes between periods
### 2. Changing probability: make sure there is a probabilty of change between states. Ex: 10% to 75% in period 2
### 3. Reversing: an account can also go back to having a "normal" aml state
BASE_TRANSITION = np.array(
    [
    [0.94, 0.05, 0.01],
    [0.10, 0.75, 0.15],
    [0.03, 0.17, 0.80],
    ],
    dtype=float,
)

NETWORK_EXPOSURE_EFFECT = np.array([-1.0, 0.25, 1.25], dtype=float)


def transition_probabilities(previous_state, account_exposure):
    """
    Calculate the probability of change for AML states taking into account the account's exposure.
    """
    ### create a base to start the calculations
    base = BASE_TRANSITION[previous_state]
    ### Adjust the the base and network exposure by penalizing the highest aml states
    adjusted = (np.log(base) + account_exposure * NETWORK_EXPOSURE_EFFECT)
    ### softmax so values are between 0 and 1
    probabilities = softmax(adjusted)
    return probabilities


def sample_next_state(previous_state, account_exposure, rng):
    """
    Select the next aml state given the previous state, network exposure and the probabilities
    This function is used in simulate_account() (equations.py) in order to assign the aml state given the orevious state info (previous_aml_state)
    """
    
    probabilities = transition_probabilities(previous_state, account_exposure)
    ### select a state between 0, 1 and 2 using the calculted probabilities
    next_aml_state = int(rng.choice(3, p=probabilities))
    return next_aml_state

