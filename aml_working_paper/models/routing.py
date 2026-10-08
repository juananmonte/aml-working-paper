import numpy as np
from scipy.special import logsumexp
import pandas as pd



def normalized_entropy(probs):
    probs = np.clip(probs, 1e-12, 1.0)

    entropy = -np.sum(
        probs * np.log(probs),
        axis=-1,
    )

    n_states = probs.shape[-1]

    return entropy / np.log(n_states)

def kl_divergence(p, q):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)

    return np.sum(
        p * (np.log(p) - np.log(q)),
        axis=-1,
    )

def posterior_normal_logpdf(x, mu, sigma):
    """
    Evaluate one observation under every posterior draw and latent state.

    x:
        (n_features,)

    mu:
        (n_samples, n_states, n_features)

    sigma:
        (n_samples, n_states, n_features)

    returns:
        (n_samples, n_states)
    """

    log_prob = (
        -0.5 * np.log(2 * np.pi)
        - np.log(sigma)
        - 0.5 * ((x[None, None, :] - mu) / sigma) ** 2
    )

    return log_prob.sum(axis=-1)

def filter_posterior_draws(sequence, pi, A, mu, sigma):
    """
    Bayesian sequential filtering for one account trajectory.

    Integrates state inference over posterior HMM parameter draws.

    Returns
    -------
    results : dict
        Contains state probabilities and routing diagnostics
        for every time step.
    """

    n_samples = pi.shape[0]
    n_states = pi.shape[1]
    T = len(sequence)

    mean_predicted = np.zeros((T, n_states))
    mean_filtered = np.zeros((T, n_states))

    uncertainty = np.zeros(T)
    expected_state_entropy = np.zeros(T)
    parameter_disagreement = np.zeros(T)

    surprise = np.zeros(T)
    information_gain = np.zeros(T)

    log_A = np.log(A)

    # Before observing X_0:
    # p(Z_0 | theta) = pi
    log_predicted = np.log(pi)

    for t in range(T):

        # ------------------------------------------------
        # 1. Belief before seeing X_t
        # ------------------------------------------------

        predicted_probs = np.exp(log_predicted)

        mean_predicted[t] = predicted_probs.mean(axis=0)

        # ------------------------------------------------
        # 2. Likelihood of X_t under every state/draw
        # ------------------------------------------------

        log_emission = posterior_normal_logpdf(
            sequence[t],
            mu,
            sigma,
        )

        log_joint = (
            log_predicted
            + log_emission
        )

        # p(X_t | X_1:t-1, theta)
        draw_log_predictive = logsumexp(
            log_joint,
            axis=1,
        )

        # ------------------------------------------------
        # 3. Bayesian predictive surprise
        # ------------------------------------------------

        log_bayesian_predictive = (
            logsumexp(draw_log_predictive)
            - np.log(n_samples)
        )

        surprise[t] = -log_bayesian_predictive

        # ------------------------------------------------
        # 4. Bayes update
        # ------------------------------------------------

        log_filtered = (
            log_joint
            - draw_log_predictive[:, None]
        )

        filtered_probs = np.exp(log_filtered)

        bayesian_filtered = filtered_probs.mean(axis=0)

        mean_filtered[t] = bayesian_filtered

        # ------------------------------------------------
        # 5. Total Bayesian state uncertainty
        # ------------------------------------------------

        uncertainty[t] = normalized_entropy(
            bayesian_filtered
        )

        # ------------------------------------------------
        # 6. Expected within-model state uncertainty
        # ------------------------------------------------

        draw_entropies = normalized_entropy(
            filtered_probs
        )

        expected_state_entropy[t] = (
            draw_entropies.mean()
        )

        # ------------------------------------------------
        # 7. Parameter disagreement
        # ------------------------------------------------
        # Eso captura cuánto desacuerdo adicional proviene de nuestra incertidumbre sobre los parámetros.
        parameter_disagreement[t] = (
            uncertainty[t]
            #Si supiera cuál de los HMMs posteriores es el correcto, ¿cuánta ambigüedad sobre el estado quedaría?
            - expected_state_entropy[t]
        )

        # ------------------------------------------------
        # 8. Information gain
        # ------------------------------------------------

        draw_ig = kl_divergence(
            filtered_probs,
            predicted_probs,
        )

        information_gain[t] = draw_ig.mean()

        # ------------------------------------------------
        # 9. Predict state at t + 1
        # ------------------------------------------------

        if t < T - 1:

            log_predicted = logsumexp(
                log_filtered[:, :, None]
                + log_A,
                axis=1,
            )

    return {
        "predicted_probs": mean_predicted,
        "filtered_probs": mean_filtered,
        "uncertainty": uncertainty,
        "expected_state_entropy": expected_state_entropy,
        "parameter_disagreement": parameter_disagreement,
        "surprise": surprise,
        "information_gain": information_gain,
    }


def extract_posterior_draws(aligned_trace):
    """
    Flatten chain and draw dimensions into a single posterior-sample dimension.
    """

    posterior = aligned_trace.posterior

    pi = posterior["pi"].values
    A = posterior["A"].values
    mu = posterior["mu"].values
    sigma = posterior["sigma"].values

    n_chains, n_draws, n_states = pi.shape
    n_features = mu.shape[-1]

    pi = pi.reshape(-1, n_states)
    A = A.reshape(-1, n_states, n_states)
    mu = mu.reshape(-1, n_states, n_features)
    sigma = sigma.reshape(-1, n_states, n_features)

    return pi, A, mu, sigma

def sequence_results_to_dataframe(
    account_id,
    results,
):

    probs = results["filtered_probs"]

    return pd.DataFrame({
        "account_id": account_id,
        "time": np.arange(len(probs)),

        "p_state_0": probs[:, 0],
        "p_state_1": probs[:, 1],
        "p_state_2": probs[:, 2],

        "uncertainty": results["uncertainty"],
        "surprise": results["surprise"],
        "information_gain": results["information_gain"],

        "expected_state_entropy":
            results["expected_state_entropy"],

        "parameter_disagreement":
            results["parameter_disagreement"],
    })

def build_routing_dataframe(
    df,
    sequences,
    pi,
    A,
    mu,
    sigma,
):
    routing_tables = []

    account_ids = sorted(df["account_id"].unique())

    assert len(account_ids) == len(sequences)

    for account_id, sequence in zip(account_ids, sequences):
        results = filter_posterior_draws(
            sequence, pi, A, mu, sigma
        )

        account_times = (
            df.loc[df["account_id"] == account_id, "time"]
            .sort_values()
            .to_numpy()
        )

        account_table = sequence_results_to_dataframe(
            account_id,
            results,
        )

        assert len(account_table) == len(account_times)
        account_table["time"] = account_times

        routing_tables.append(account_table)

    return pd.concat(routing_tables, ignore_index=True)