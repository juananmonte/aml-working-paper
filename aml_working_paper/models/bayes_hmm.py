import pymc as pm
import numpy as np
import pytensor.tensor as pt
from pytensor.scan import scan
#from scipy.stats import norm


def pdf_normal_hmm(sequences, mu, sigma):
    """
    Get the probability density function for a normal distribution in order to check if a data point is compatible with the distribution
    sequence:
    mu:
    sigma
    """
    ### Pymc uses pytensor whic is faster than scipy or numpy
    normal_dist = pm.Normal.dist(mu=mu, sigma=sigma)
    ### I use [..., None, :] because sequences have a "time" layer and it needs to match the shape of mu and sigma. It also helps with broadcasting.
    log_prob = pm.logp(normal_dist, sequences[..., None, :])
    ### use pytensor to get the sum since pymc doesn't have this function
    emissions_hmm = pm.math.sum(log_prob, axis=-1)

    return emissions_hmm

def forward_step(sequences, previous, current, mu, sigma):
    """
    Calculate how much evidence in favour, for a moment t, each hidden state has

    sequences, log_prev, log_current, mu, sigma
    """

    emissions_hmm = pdf_normal_hmm(sequences, mu, sigma)
    ### np.newaxis used in order to put the shapes in the same dimentions
    transition_combinations = (previous[:, :, np.newaxis] + current[np.newaxis, :, :])
    ### add all the contributions of all the combinations from the previous states
    predicted_state = pm.math.logsumexp(transition_combinations, axis=1)
    
    return (predicted_state + emissions_hmm)

def forward_log_likelihood(sequences, pi, current, mu, sigma):
    """
    calculate te forward step, move to the remaining periods and calculate the total log likelihood

    sequences, log_prev, log_current, mu, sigma
    """

    log_pi = pm.math.log(pi)
    log_A = pm.math.log(current)

    ### get the first period/state data
    first_observation = sequences[:, 0, :]
    first_log_emission = pdf_normal_hmm(first_observation, mu, sigma)
    log_alpha_initial = log_pi[None, :] + first_log_emission

    ### 1 because the period 0 is already taken as it is the base
    remaining_observations = sequences[:, 1:, :]

    ### dimshufle is used to put the time axis (sequences) first for the scan method
    observations_by_time = remaining_observations.dimshuffle(1, 0, 2)

    ### scan is a for loop function for operations with pytensors/pymc. 
    ### In this case, it builds a table of results for each period ang gets the forward step for each one. With that forward step, we can calculate the next period.
    log_alpha_history, _ = scan(fn=forward_step, sequences=[observations_by_time], outputs_info=[log_alpha_initial], non_sequences=[log_A, mu, sigma])
    final_log_alpha = log_alpha_history[-1]
    ###logsumexp get the total values of a sequence for an account
    account_log_likelihood = pm.math.logsumexp(final_log_alpha,axis=1)
    ### now, we add the total values for all accounts
    total_log_likelihood = pm.math.sum(account_log_likelihood)

    return total_log_likelihood

def build_hmm_model(sequences):

    """
    Build the model defining the starting points for the Hidden Markov Models

    """
    
    data = np.stack(sequences).astype("float64")
    n_states = 3
    n_features = data.shape[2]

    with pm.Model() as model:
        initial_probs = pm.Dirichlet("pi", a=np.ones(n_states))
        transition_matrix = pm.Dirichlet("current", a=np.ones((n_states, n_states)), shape=(n_states, n_states))
        state_means = pm.Normal("mu", mu=0, sigma=2,shape=(n_states, n_features))
        state_stds = pm.HalfNormal("sigma", sigma=1, shape=(n_states, n_features))
        observations = pt.as_tensor_variable(data)
        log_likelihood = forward_log_likelihood(observations, initial_probs, transition_matrix, state_means, state_stds)
        pm.Potential("hmm_likelihood",log_likelihood)

    return model