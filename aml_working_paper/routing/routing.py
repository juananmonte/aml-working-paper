
import numpy as np
import pandas as pd
import pymc as pm
###from scipy.stats import entropy

def normalized_entropy(probabilities):
    """
    How uncertain is the Markov model about the current state of an account BETWEEN their values
    probabilities: Probability distribution of the hidden states for the current account
    """
    ### Clip is used in order to remove probabilities that might be too small and python could assign it as inf. Also, the max value being 1.
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    ### Shannon entropy (pymc is faster than scipy's entropy)
    ### entropy = entropy(probabilities, axis=0)
    ### https://discourse.pymc.io/t/compute-shannon-entropy-with-pymc4/10663
    shannon_entropy = -np.sum(probabilities * np.log(probabilities), axis=-1)
    number_states = probabilities.shape[-1]
    normal_shannon = shannon_entropy / np.log(number_states)
    return normal_shannon


def divergence_KL(state_before, state_after):
    """
    How uncertain is the Markov model about the difference BETWEEN previous and after probability distributions
    probabilities: 
    """
    state_before = np.clip(state_before, 1e-8, 1.0)
    state_after = np.clip(state_after, 1e-8, 1.0)
    divergence = state_before * np.log(state_before / state_after)
    states_contibutions = np.sum(divergence, axis=-1)
    return states_contibutions

def posterior_normal_logpdf(x, mu, sigma):
    """
    Esta función te resultará familiar porque se parece muchísimo a normal_logpdf_batch_pt() de hmm_bayes.py. Sin embargo, tiene una diferencia fundamental:
    Antes calculábamos la log-densidad para cada cuenta y estado. Ahora la calculamos para cada muestra posterior y estado.
    Esto nos permite incorporar la incertidumbre sobre los parámetros que estimamos con PyMC.
    """

    constant = -0.5 * np.log(2 * np.pi)
    standardized_distance = (x[None, None, :] - mu) / sigma
    log_density = ((constant) - (np.log(sigma)) - (0.5 * (standardized_distance**2)))
    return log_density.sum(axis=-1)


def filter_posterior_draws(sequence, pi, current, mu, sigma):
    """
    ¿Cómo cambian nuestras creencias sobre el estado AML de una cuenta cuando observamos sus transacciones a través del tiempo?
    """
    no_samples, no_states = pi.shape
    no_periods = len(sequence)

    predicted_history = np.zeros((no_periods, no_states))
    filtered_history = np.zeros((no_periods, no_states))
    uncertainty = np.zeros(no_periods)
    surprise = np.zeros(no_periods)
    information_gain = np.zeros(no_periods)
    expected_state_entropy = np.zeros(no_periods)
    parameter_disagreement = np.zeros(no_periods)

    log_predicted = np.log(pi)
    log_A = np.log(current)

    for i in range(no_periods):

        predicted = np.exp(log_predicted)
        predicted_history[i] = predicted.mean(axis=0)

        log_emission = posterior_normal_logpdf(sequence[i], mu, sigma)
        log_joint = log_predicted + log_emission
        log_evidence = pm.math.logsumexp(log_joint, axis=1)
        log_predictive = pm.math.logsumexp((log_evidence) - np.log(no_samples))
        log_filtered = log_joint - log_evidence[:, None]

        surprise[i] = -log_predictive

        filtered = np.exp(log_filtered)
        mean_filtered = filtered.mean(axis=0)
        filtered_history[i] = mean_filtered

        uncertainty[i] = normalized_entropy(mean_filtered)

        entropy_per_sample = normalized_entropy(filtered)
        expected_state_entropy[i] = entropy_per_sample.mean()
        parameter_disagreement[i] = (uncertainty[i] - expected_state_entropy[i])
        information_gain[i] = divergence_KL(filtered, predicted).mean()

        if i < no_periods - 1:
            log_predicted = pm.math.logsumexp(log_filtered[:, :, None] + log_A, axis=1)

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

    posterior = aligned_trace.posterior

    initial_probs = posterior["pi"].values
    transitions = posterior["A"].values
    state_means = posterior["mu"].values
    state_stds = posterior["sigma"].values

    no_chains, no_draws, no_states = initial_probs.shape
    n_features = state_means.shape[-1]
    no_samples = no_chains * no_draws

    initial_probs = initial_probs.reshape(no_samples, no_states)
    transitions = transitions.reshape(no_samples, no_states, no_states)

    state_means = state_means.reshape(no_samples, no_states, n_features)
    state_stds = state_stds.reshape(no_samples, no_states, n_features)

    return initial_probs, transitions, state_means, state_stds

def sequence_results_to_dataframe(account_id, results):

    probabilities = results["filtered_probabilities"]
    no_periods = len(probabilities)

    table = pd.DataFrame({
        "account_id": account_id,
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

    tables = []
    account_ids = sorted(df["account_id"].unique())

    for account_id, sequence in zip(account_ids, sequences):
        results = filter_posterior_draws(sequence, pi, current, mu, sigma)
        times = (df.loc[df["account_id"] == account_id, "time"].sort_values().to_numpy())
        account_table = sequence_results_to_dataframe(account_id, results)
        account_table["time"] = times
        tables.append(account_table)

    return pd.concat(tables, ignore_index=True)
