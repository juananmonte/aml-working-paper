import numpy as np
from scipy.special import logsumexp


def normal_logpdf(x, mu, sigma):
    log_prob = (
        -0.5 * np.log(2 * np.pi)
        - np.log(sigma)
        - 0.5 * ((x - mu) / sigma) ** 2
    )

    return log_prob.sum(axis=1)

def filter_sequence(sequence, pi, A, mu, sigma):
    """
    Sequential filtering for one account trajectory
    and one posterior parameter draw.

    Returns
    -------
    predicted_probs : array, shape (T, K)
        p(Z_t | X_1:t-1)

    filtered_probs : array, shape (T, K)
        p(Z_t | X_1:t)

    log_predictive : array, shape (T,)
        log p(X_t | X_1:t-1)
    """

    T = len(sequence)
    K = len(pi)

    predicted_probs = np.zeros((T, K))
    filtered_probs = np.zeros((T, K))
    log_predictive = np.zeros(T)

    log_pi = np.log(pi)
    log_A = np.log(A)

    # Initial state prediction:
    # p(Z_1)
    log_predicted = log_pi

    for t in range(T):

        # How well does each state explain X_t?
        log_emission = normal_logpdf(
            sequence[t],
            mu,
            sigma,
        )

        # Predictive probability of observing X_t
        log_joint = log_predicted + log_emission

        log_predictive[t] = logsumexp(log_joint)

        # Bayes update
        log_filtered = (
            log_joint
            - log_predictive[t]
        )

        predicted_probs[t] = np.exp(log_predicted)
        filtered_probs[t] = np.exp(log_filtered)

        # Predict next state
        if t < T - 1:
            log_predicted = logsumexp(
                log_filtered[:, None] + log_A,
                axis=0,
            )

    return (
        predicted_probs,
        filtered_probs,
        log_predictive,
    )

def normalized_entropy(probs):
    """
    Normalized Shannon entropy.

    Returns values between 0 and 1.
    """

    probs = np.clip(probs, 1e-12, 1.0)

    entropy = -np.sum(
        probs * np.log(probs),
        axis=-1,
    )

    K = probs.shape[-1]

    return entropy / np.log(K)

def predictive_surprise(log_predictive):
    return -log_predictive

def information_gain(predicted_probs, filtered_probs):
    predicted_probs = np.clip(
        predicted_probs,
        1e-12,
        1.0,
    )

    filtered_probs = np.clip(
        filtered_probs,
        1e-12,
        1.0,
    )

    return np.sum(
        filtered_probs
        * (
            np.log(filtered_probs)
            - np.log(predicted_probs)
        ),
        axis=-1,
    )