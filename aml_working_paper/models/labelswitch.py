
import copy
import itertools
import numpy as np


def labelswitch(data):

    # Crear una copia para conservar los resultados originales
    aligned = copy.deepcopy(data)
    posterior = data.posterior

    pi = posterior["pi"].values
    A = posterior["A"].values
    mu = posterior["mu"].values
    sigma = posterior["sigma"].values

    n_chains, n_draws, n_states, _ = mu.shape

    # Utilizar la cadena 0 como referencia
    reference = mu[0].mean(axis=0)

    # Generar todas las permutaciones posibles
    permutations = list(itertools.permutations(range(n_states)))

    # Guardar la permutación elegida para cada muestra
    selected = np.zeros((n_chains, n_draws, n_states), dtype=int)

    # Revisar cada muestra de cada cadena
    for chain in range(n_chains):
        for draw in range(n_draws):

            current_mu = mu[chain, draw]

            best_permutation = None
            lowest_error = float("inf")

            # Buscar el orden más parecido a la referencia
            for permutation in permutations:

                error = np.sum((current_mu[list(permutation)] - reference) ** 2)

                if error < lowest_error:
                    lowest_error = error
                    best_permutation = list(permutation)

            selected[chain, draw] = best_permutation

            # Reordenar los parámetros en la copia
            aligned.posterior["pi"].values[chain, draw] = (pi[chain, draw, best_permutation])
            aligned.posterior["mu"].values[chain, draw] = (mu[chain, draw, best_permutation, :])
            aligned.posterior["sigma"].values[chain, draw] = (sigma[chain, draw, best_permutation, :])

            # Reordenar filas y columnas de la matriz de transición
            aligned.posterior["A"].values[chain, draw] = (A[chain, draw][np.ix_(best_permutation, best_permutation)])

    return aligned, selected
