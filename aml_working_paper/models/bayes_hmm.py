import pymc as pm
import numpy as np
import pytensor.tensor as pt
from pytensor.scan import scan


def normal_logpdf_batch_pt(x, mu, sigma):
    """
    x:     (n_accounts, n_features)
    mu:    (n_states, n_features)
    sigma: (n_states, n_features)

    returns:
        (n_accounts, n_states)
    """
    log_prob = (-0.5 * pt.log(2 * np.pi) - pt.log(sigma[None, :, :]) - 0.5 * ( (x[:, None, :] - mu[None, :, :]) / sigma[None, :, :]) ** 2)
    return pt.sum(log_prob, axis=2)


def batch_forward_log_likelihood_pt(sequences, pi, A, mu, sigma):
    """
    Forward likelihood for all account trajectories simultaneously.

    sequences:
        (n_accounts, n_time, n_features)
    """

    log_pi = pt.log(pi)
    log_A = pt.log(A)

    # First observation for every account
    log_emission_0 = normal_logpdf_batch_pt(
        sequences[:, 0, :],
        mu,
        sigma,
    )

    log_alpha_0 = (
        log_pi[None, :]
        + log_emission_0
    )

    # scan expects time as the first dimension
    observations = sequences[:, 1:, :].dimshuffle(1, 0, 2)

    def forward_step(x_t, log_alpha_prev, log_A, mu, sigma):

        # (account, previous_state, next_state)
        log_transition = (
            log_alpha_prev[:, :, None]
            + log_A[None, :, :]
        )

        # Sum over possible previous states
        log_predicted_state = pt.logsumexp(
            log_transition,
            axis=1,
        )

        log_emission = normal_logpdf_batch_pt(
            x_t,
            mu,
            sigma,
        )

        return (
            log_predicted_state
            + log_emission
        )

    log_alphas, _ = scan(
        fn=forward_step,
        sequences=[observations],
        outputs_info=[log_alpha_0],
        non_sequences=[log_A, mu, sigma],
    )

    # Last alpha for every account
    final_log_alpha = log_alphas[-1]

    # Likelihood of each account
    account_log_likelihood = pt.logsumexp(
        final_log_alpha,
        axis=1,
    )

    # Total likelihood
    return pt.sum(account_log_likelihood)

def build_hmm_model(train_sequences):

    train_array = np.stack(train_sequences).astype("float64")
    n_states = 3
    n_features = train_array.shape[2]

    with pm.Model() as model:

        pi = pm.Dirichlet("pi", a=np.ones(n_states))
        A = pm.Dirichlet("A", a=np.ones((n_states, n_states)), shape=(n_states, n_states))
        mu = pm.Normal("mu", mu=0.0, sigma=2.0, shape=(n_states, n_features))
        sigma = pm.HalfNormal("sigma", sigma=1.0, shape=(n_states, n_features))
        sequences = pt.as_tensor_variable(train_array)
        log_likelihood = batch_forward_log_likelihood_pt(sequences, pi, A, mu, sigma)
        pm.Potential("hmm_likelihood", log_likelihood)

    return model