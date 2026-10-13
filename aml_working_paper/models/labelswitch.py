
import copy
import itertools
import numpy as np


def labelswitch(data):

    aligned = copy.deepcopy(data)
    posterior = data.posterior

    pi = posterior["pi"].values
    current = posterior["A"].values
    mu = posterior["mu"].values
    sigma = posterior["sigma"].values

    no_chains, no_draws, no_states, _ = mu.shape
    reference = mu[0].mean(axis=0)

    permutations = list(itertools.permutations(range(no_states)))
    selected = np.zeros((no_chains, no_draws, no_states), dtype=int)

    for chain in range(no_chains):
        for draw in range(no_draws):

            current_mu = mu[chain, draw]

            best_permutation = None
            lowest_error = float("inf")

            for permutation in permutations:
                error = np.sum((current_mu[list(permutation)] - reference) ** 2)
                if error < lowest_error:
                    lowest_error = error
                    best_permutation = list(permutation)

            selected[chain, draw] = best_permutation

            aligned.posterior["pi"].values[chain, draw] = (pi[chain, draw, best_permutation])
            aligned.posterior["mu"].values[chain, draw] = (mu[chain, draw, best_permutation, :])
            aligned.posterior["sigma"].values[chain, draw] = (sigma[chain, draw, best_permutation, :])


            aligned.posterior["current"].values[chain, draw] = (current[chain, draw][np.ix_(best_permutation, best_permutation)])

    return aligned, selected
