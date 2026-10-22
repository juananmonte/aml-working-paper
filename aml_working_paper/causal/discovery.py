
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.search.ConstraintBased.FCI import fci


def discover_pc(data, features):
    """Run an exploratory PC causal discovery baseline."""

    # Select only the observable variables
    observations = data[features].to_numpy()
    # Run PC using Fisher's Z conditional independence test
    graph = pc(observations, alpha=0.05, indep_test=fisherz, show_progress=False,)

    return graph


def discover_fci(data, features):
    """Run an exploratory FCI causal discovery baseline."""

    # Select observable variables
    observations = data[features].to_numpy()
    # Run FCI
    graph, edges = fci(observations, independence_test_method="fisherz", alpha=0.05, verbose=False, show_progress=False,)

    return graph
