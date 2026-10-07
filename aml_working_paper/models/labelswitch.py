import copy
import itertools
import numpy as np


def relabel_posterior_states(idata):
    """
    Align latent-state labels across posterior draws.

    The alignment is based on the state-specific emission means (mu).
    No ground-truth aml_state information is used.

    Parameters
    ----------
    idata : arviz.InferenceData
        PyMC posterior samples.

    Returns
    -------
    aligned_idata : arviz.InferenceData
        Copy of the posterior with consistent state labels.

    permutations : np.ndarray
        Permutation selected for every chain and draw.
        Shape: (n_chains, n_draws, n_states)
    """

    #¿Qué hace? Primero, hace una copia limpia de tus datos para no arruinar el original (deepcopy). Luego saca los parámetros del HMM de cada iteración del muestreo:
    #las probabilidades iniciales (π), las transiciones (A), las medias (μ) y las desviaciones estándar (σ).

    aligned_idata = copy.deepcopy(idata)
    posterior = idata.posterior
    pi = posterior["pi"].values
    A = posterior["A"].values
    mu = posterior["mu"].values
    sigma = posterior["sigma"].values

    # Expected shapes:
    # pi    -> (chain, draw, state)
    # A     -> (chain, draw, state, state)
    # mu    -> (chain, draw, state, feature)
    # sigma -> (chain, draw, state, feature)

    n_chains, n_draws, n_states, _ = mu.shape

    # Use the mean state profiles from chain 0 as our arbitrary reference.

    #La clave de la función: Como el modelo no sabe cuál es el orden "correcto", la función decide usar la cadena 0 como su modelo de referencia. 
    #Saca el promedio de las medias (μ) de esa cadena para usarlo como el "molde" ideal. El objetivo de todo el script será hacer que todas 
    #las demás iteraciones se parezcan a este molde.
    reference_mu = mu[0].mean(axis=0)

    #¿Qué hace? Si tienes 3 estados (0, 1, 2), itertools.permutations genera todas las formas posibles de ordenarlos. 
    #Hay 6 combinaciones en total (por ejemplo: [0,1,2], [1,0,2], [2,1,0], etc.). 
    #La función va a probar cuál de estas combinaciones encaja mejor con nuestro molde.

    possible_permutations = list(itertools.permutations(range(n_states)))

    pi_aligned = np.empty_like(pi)
    A_aligned = np.empty_like(A)
    mu_aligned = np.empty_like(mu)
    sigma_aligned = np.empty_like(sigma)

    selected_permutations = np.empty((n_chains, n_draws, n_states), dtype=int)


    #Para cada iteración específica, extrae las medias actuales (current_mu) y hace un concurso para ver qué permutación tiene el menor 
    # error matemático comparado con el molde (reference_mu):
    for chain in range(n_chains):
        for draw in range(n_draws):

            current_mu = mu[chain, draw]

            best_perm = None
            best_cost = np.inf

            #En lenguaje sencillo: Prueba la combinación [1,0,2]. ¿Se parece al molde? Calcula la resta al cuadrado (el cost). 
            #Si esa resta da un número muy pequeño, significa que esa permutación es la que corrige el cruce de cables en esa iteración.
            for perm in possible_permutations:
                perm = np.array(perm)

                candidate_mu = current_mu[perm]

                cost = np.sum(
                    (candidate_mu - reference_mu) ** 2
                )

                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm

            selected_permutations[chain, draw] = best_perm

            # State-level parameters
            #Una vez descubierta la combinación ganadora (best_perm) para esa iteración, la función reordena todos los parámetros para que coincidan:
            pi_aligned[chain, draw] = (
                pi[chain, draw, best_perm]
            )

            mu_aligned[chain, draw] = (
                mu[chain, draw, best_perm, :]
            )

            sigma_aligned[chain, draw] = (
                sigma[chain, draw, best_perm, :]
            )

            # Transition matrix:
            # permute both previous-state rows
            # and next-state columns.
            # La matriz A indica la probabilidad de pasar del estado i al estado j. 
            # Como es una matriz bidimensional (filas y columnas), no basta con ordenar las filas. 
            # np.ix_ es un truco de NumPy que ordena tanto las filas como las columnas al mismo tiempo para que la matriz de transiciones 
            # siga teniendo sentido matemático.
            A_aligned[chain, draw] = (
                A[chain, draw][np.ix_(best_perm, best_perm)]
            )

    # Finalmente, reemplaza los datos originales caóticos con los nuevos datos alineados dentro del objeto de ArviZ (aligned_idata) 
    # y te lo devuelve listo para graficar, junto con una lista de qué permutación se usó en cada momento por si quieres analizarla.

    aligned_idata.posterior["pi"].values = pi_aligned
    aligned_idata.posterior["A"].values = A_aligned
    aligned_idata.posterior["mu"].values = mu_aligned
    aligned_idata.posterior["sigma"].values = sigma_aligned

    return aligned_idata, selected_permutations