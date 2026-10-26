
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz


def discover_pc(data, features, alpha=0.05):
    """Run an exploratory PC causal discovery baseline."""

    observations = data[features].to_numpy()

    graph = pc(
        observations,
        alpha=alpha,
        indep_test=fisherz,
        show_progress=False,
    )

    return graph


def discover_fci(data, features, alpha=0.05):
    """Run an exploratory FCI causal discovery baseline."""

    observations = data[features].to_numpy()

    graph, edges = fci(
        observations,
        independence_test_method="fisherz",
        alpha=alpha,
        verbose=False,
        show_progress=False,
    )

    return graph
