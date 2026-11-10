import pymc as pm
import numpy as np
import pytensor.tensor as pt
from pytensor.scan import scan
#from scipy.stats import norm


def emission_log_prob(sequences, mu, sigma):
    """
    Get the probability density function for a normal distribution in order to check if a data point is compatible with the distribution
    sequence.
    Note: A supossition is made regarding how the observed variables are conditionally independent given the hidden state. 
    """
    ###Get the normal distribution for each state and variable. Pymc uses pytensor whic is faster than scipy or numpy. 
    normal_dist = pm.Normal.dist(mu=mu, sigma=sigma)
    ### get log density of the observations. This helps with possible small values.
    ### [..., None, :] is used because the sequences have a "time" layer (accounts, periods, variables) and it needs to match the shape of mu and sigma.
    log_prob = pm.logp(normal_dist, sequences[..., None, :])
    # Combine the contributions of all observed features and retunr the shape (accounts, periods, hidden states) needed
    emissions_hmm = pm.math.sum(log_prob, axis=-1)
    return emissions_hmm

def forward_step(sequences, previous, current, mu, sigma):
    """
    Since the previous aml state is hidden, there is a need to accumulate and ponderate all the information from all previous states
    Take the observations of all accounts for one period.
    """
    ### how the observed columns/variables behave under different aml states
    emissions_hmm = emission_log_prob(sequences, mu, sigma)
    ### get all the combinations of the probabilities from previous aml states and the current probabilities. Ex: aml-state 0 -> aml-state 2, aml-state 1 -> aml-state 2, aml-state 2 -> aml-state 2
    ### np.newaxis used in order to put the shapes in the same dimentions
    ### current acts 
    transition_combinations = (previous[:, :, np.newaxis] + current[np.newaxis, :, :])
    ### add all the contributions from all the possible combinations from the previous states. 
    predicted_state = pm.math.logsumexp(transition_combinations, axis=1)
    ### Add the current prediction probabilities and all the accumulated evidence/information the model has before.
    ### This will allow the model to update it's parameters
    return (predicted_state + emissions_hmm)

def forward_log_likelihood(sequences, pi, current, mu, sigma):
    """
    calculate the forward step, move to the remaining periods and calculate the total log likelihood
    """
    ### initial base to calculate emissions
    ### make them logs since forward_step and emission_log_prob are in log
    log_pi = pm.math.log(pi)

    ### get the first period data for all accounts
    first_observation = sequences[:, 0, :]
    ### Evaluate the first observation under each hidden state
    first_log_emission = emission_log_prob(first_observation, mu, sigma)
    ### Initialize the Forward algorithm using initial state probabilities
    log_alpha_initial = log_pi[None, :] + first_log_emission

    ### 1 because the period 0 is already taken as it is the base
    remaining_observations = sequences[:, 1:, :]

    ### dimshufle is used to put the observations with the period axis first. This is because of Forward Step: it takes the first period of all accounts, then the second period and so on
    observations_by_time = remaining_observations.dimshuffle(1, 0, 2)
    
    ### probability of moving from one state to another
    log_current = pm.math.log(current)

    ### scan is a for loop function that deals with pytensors/pymc. 
    ### In this case, it builds a table of results for each period ang gets the forward step for each one. With that forward step, we can calculate the next period.
    log_alpha_history, _ = scan(fn=forward_step, sequences=[observations_by_time], outputs_info=[log_alpha_initial], non_sequences=[log_current, mu, sigma])
    ### get the final values
    final_log_alpha = log_alpha_history[-1]
    ###logsumexp get the total values of a sequence for an account
    account_log_likelihood = pm.math.logsumexp(final_log_alpha,axis=1)
    ### now, we add the total values for all accounts
    total_log_likelihood = pm.math.sum(account_log_likelihood)

    return total_log_likelihood

def build_hmm_model(sequences):

    """
    Build the model defining the starting points for the Hidden Markov Models. 

    """
    
    data = np.stack(sequences).astype("float64")
    n_states = 3
    n_features = data.shape[2]

    with pm.Model() as model:
        ### using Dricjlet, create a probability distribution that, when added, gives 1 for both the initial probabilitie (of the three possible aml state)
        ### and the transition matrix (the change from one state to another for each aml state)
        initial_probs = pm.Dirichlet("pi", a=np.ones(n_states))
        transition_matrix = pm.Dirichlet("current", a=np.ones((n_states, n_states)), shape=(n_states, n_states))
        ### Normal distribution of mu to get uncertainty
        state_means = pm.Normal("mu", mu=0, sigma=2,shape=(n_states, n_features))
        ### Half normal is used so there are no negative values 
        state_stds = pm.HalfNormal("sigma", sigma=1, shape=(n_states, n_features))
        observations = pt.as_tensor_variable(data)
        ### Get how compatible is a set of dats with the HMM parameters
        log_likelihood = forward_log_likelihood(observations, initial_probs, transition_matrix, state_means, state_stds)
        ### add the log likelihood and update priors
        pm.Potential("hmm_likelihood",log_likelihood)

    return model