
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import logsumexp, rel_entr

### Shannon entropy (pymc is faster than scipy's entropy)
### entropy = entropy(probabilities, axis=0)
### https://discourse.pymc.io/t/compute-shannon-entropy-with-pymc4/10663

def normalized_entropy(probabilities):
    """
    How uncertain is the HMM about the current aml state of an account given their probabilities
    Shannon entropy goes up when the probabilities between the three aml states are too similar. 
    """
    ### Clip is used in order to remove probabilities that might be too small and python could assign as inf. Ex log(0).
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    ### Shannon entropy is used because there could be cases where the probabilities for all 3 states are too similar.
    shannon_entropy = -np.sum(probabilities * np.log(probabilities), axis=-1)
    number_states = probabilities.shape[-1]
    ### normalize entropy so it is easier to interpret
    normal_shannon = shannon_entropy / np.log(number_states)
    return normal_shannon


def divergence_KL(state_before, state_after):
    """
    Get the Kullback-Leibler divergence
    Measure the change in probabilities of the AML states when new information (transactions) is taken into account.
    The higher the change, the higher the divergence. Therefore, the HMM updates its beliefs. 
    """
    ### Filter out really small values
    state_before = np.clip(state_before, 1e-8, 1.0)
    state_after = np.clip(state_after, 1e-8, 1.0)
    ### calculate the divergence...
    divergence = rel_entr(state_before, state_after).sum(axis=-1)
    ### ...and add the divergence for all aml states
    states_contibutions = np.sum(divergence, axis=-1)
    return states_contibutions

def normal_dist_in_log(x, mu, sigma):
    """
    How compatible is a single observation's characteristic against each one of the three AML states distributions for that characteristic
    Calculate the log density of a normal distribution in order to get the log emissions
    """
    ### the logarithmic representation for the constant term in the normal distribution formula
    constant = -0.5 * np.log(2 * np.pi)
    ### how far is the characteristic from the estimated mean in deviation standard
    standardized_distance = (x[None, None, :] - mu) / sigma
    log_density = ((constant) - (np.log(sigma)) - (0.5 * (standardized_distance**2)))
    log_density = log_density.sum(axis=-1)
    return log_density


def filter_posterior_draws(sequence, pi, current, mu, sigma):
    """
    Measure how much the model's beliefs about an AML state for an account change when cheking that account transactions over time.
    check the previous probabilities
    how do those probabilities change after seeing them
    how unsure is the model
    how surprising was the observation
    how much did the beliefs change
    """
    ### get the shapes in order to store the results correclty and base probabilities (pi)
    no_samples, no_states = pi.shape
    no_periods = len(sequence)

    ### create empty varaibles to fill them in the iteration with their number of period and the aml_states
    predicted_history = np.zeros((no_periods, no_states))
    filtered_history = np.zeros((no_periods, no_states))
    uncertainty = np.zeros(no_periods)
    surprise = np.zeros(no_periods)
    information_gain = np.zeros(no_periods)
    expected_state_entropy = np.zeros(no_periods)
    parameter_disagreement = np.zeros(no_periods)

    ### define a base distribuion for the hidden states.
    log_predicted = np.log(pi)
    ### the transitionprobabilities between the three aml states
    ### current contains the present transition matrices
    log_A = np.log(current)

    ### bayesian update
    for i in range(no_periods):

        ### exp is used to save the mean of the probabilities for each aml state in order to check the variability
        ### betwwen the posterior parameters
        predicted = np.exp(log_predicted)
        predicted_history[i] = predicted.mean(axis=0)

        ### get log emission to check thecompatibility of the observation and the current aml state
        log_emission = normal_dist_in_log(sequence[i], mu, sigma)
        ### add the compatibility of the observation with the prediction made
        log_joint = log_predicted + log_emission
        ### normalizes log joint so values are between 0and 1
        log_evidence = logsumexp(log_joint, axis=1)
        ### get the log of the mean fo the predicted densities (log evidence)
        log_predictive = logsumexp(log_evidence - np.log(no_samples))
        ### Get updated probabilities of the aml states
        log_filtered = log_joint - log_evidence[:, None]

        ### if an observations has a higher density then it is less surprising. And viceversa. 
        surprise[i] = -log_predictive

        ### get the mean for each of the aml states
        filtered = np.exp(log_filtered)
        mean_filtered = filtered.mean(axis=0)
        filtered_history[i] = mean_filtered

        ### use shannon entropy for the means
        uncertainty[i] = normalized_entropy(mean_filtered)

        ### ge the uncertainty of the predcited aml states for each sample and then get the mean of it. 
        entropy_per_sample = normalized_entropy(filtered)
        expected_state_entropy[i] = entropy_per_sample.mean()

        ### how much of the expected entropy is different from the overall uncertainty
        parameter_disagreement[i] = (uncertainty[i] - expected_state_entropy[i])
        information_gain[i] = divergence_KL(filtered, predicted).mean()


        ### Hidden Markov model prediction step
        if i < no_periods - 1:
            log_predicted = logsumexp(log_filtered[:, :, None] + log_A,axis=1)

    return {
        "predicted_probabilities": predicted_history,
        "filtered_probabilities": filtered_history,
        "uncertainty": uncertainty,
        "surprise": surprise,
        "information_gain": information_gain,
        "expected_state_entropy": expected_state_entropy,
        "parameter_disagreement": parameter_disagreement,
    }


def extract_posterior_draws(aligned_trace):

    """
    Use the obtained probabilities from the current period and predict the next periods AML states
    """

    ### get the switched labels from labelswitch.py
    posterior = aligned_trace.posterior
    ### get the estimated values from the model
    initial_probs = posterior["pi"].values
    transitions = posterior["current"].values
    state_means = posterior["mu"].values
    state_stds = posterior["sigma"].values
    ### get the dimentions used to train the model Ex: 4 chains, 200 samples, 3 aml states and 3 observed variables
    no_chains, no_draws, no_states = initial_probs.shape
    n_features = state_means.shape[-1]
    no_samples = no_chains * no_draws

    ### put chains and samples into one dimention. This is done so there are no unpack problems with filter posterior draws
    ### in build_routing_dataframe because of pi. 
    initial_probs = initial_probs.reshape(no_samples, no_states)
    transitions = transitions.reshape(no_samples, no_states, no_states)
    state_means = state_means.reshape(no_samples, no_states, n_features)
    state_stds = state_stds.reshape(no_samples, no_states, n_features)

    return initial_probs, transitions, state_means, state_stds

def sequence_results_to_dataframe(account_id, results):
    """
    Transform filter_posterior_draws dictionary into a df
    """

    probabilities = results["filtered_probabilities"]
    no_periods = len(probabilities)

    table = pd.DataFrame({
        "account_id": account_id,
        ### order time values so they match the original ones
        "time": np.arange(no_periods),
        "p_state_0": probabilities[:, 0],
        "p_state_1": probabilities[:, 1],
        "p_state_2": probabilities[:, 2],
        "uncertainty": results["uncertainty"],
        "surprise": results["surprise"],
        "information_gain": results["information_gain"],
        "expected_state_entropy": results["expected_state_entropy"],
        "parameter_disagreement": results["parameter_disagreement"],
        })

    return table

def build_routing_dataframe(df, sequences, pi, current, mu, sigma):

    """
    Takes allthe accounts in the df from filter_posterior_draws and calculates the probabilities and uncertainty
    Puts them into one dataframe
    """

    tables = []
    account_ids = sorted(df["account_id"].unique())

    for account_id, sequence in zip(account_ids, sequences):
        results = filter_posterior_draws(sequence, pi, current, mu, sigma)
        times = (df.loc[df["account_id"] == account_id, "time"].sort_values().to_numpy())
        account_table = sequence_results_to_dataframe(account_id, results)
        account_table["time"] = times
        tables.append(account_table)

    routing_df = pd.concat(tables, ignore_index=True)

    return routing_df
