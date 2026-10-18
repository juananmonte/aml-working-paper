
import copy
import itertools
import numpy as np


def labelswitch(data):

    """
    Function that calculates the cuadratic error between different permutations. 
    The one with the smaller error has the correct order of the labels 
    """

    ### make a copy of the data to not ruin the original
    aligned = copy.deepcopy(data)
    posterior = data.posterior

    ### Get pi, current, sigma and mu for, when the correct permution is found, the values are assigned to the correct label
    pi = posterior["pi"].values
    current = posterior["current"].values
    mu = posterior["mu"].values
    sigma = posterior["sigma"].values

    ### Use the first mu as the base reference for checking the order of the the labels
    no_chains, no_draws, no_states, _ = mu.shape
    reference = mu[0].mean(axis=0)

    ### Get the permutation for the states. in this case 3!=6. So: [0,1,2], [0,2,1], [1,0,2] ...
    permutations = list(itertools.permutations(range(no_states)))
    ###create an empty matrix with the number of draws, the number of states (3) and the observed variables.
    selected = np.zeros((no_chains, no_draws, no_states), dtype=int)

    ### for each of the chains obtained...
    for chain in range(no_chains):
        ### for each of the parameters obtained on a draw...
        for draw in range(no_draws):
            #...get the chain it was obtained from and the mu value on tha draw
            current_mu = mu[chain, draw]
            best_permutation = None
            lowest_error = float("inf")

            ### for every permuation...
            for permutation in permutations:
                ###calculate the quadratic error between the mu from the current estimated and the mu used for reference (reference = mu[0].mean(axis=0))
                error = np.sum((current_mu[list(permutation)] - reference) ** 2)
                ### fill the values to get the lowest error and the best permutation
                if error < lowest_error:
                    lowest_error = error
                    best_permutation = list(permutation)

            ### fill the matrix with the correct parameters to keep having the same HMM
            selected[chain, draw] = best_permutation

            ### do the same steps for "current_mu" but for the other parameters
            aligned.posterior["pi"].values[chain, draw] = (pi[chain, draw, best_permutation])
            aligned.posterior["mu"].values[chain, draw] = (mu[chain, draw, best_permutation, :])
            aligned.posterior["sigma"].values[chain, draw] = (sigma[chain, draw, best_permutation, :])
            aligned.posterior["current"].values[chain, draw] = (current[chain, draw][np.ix_(best_permutation, best_permutation)])

    return aligned, selected
