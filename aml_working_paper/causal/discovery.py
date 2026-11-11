
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz


def discover_pc(data, features, alpha=0.05):
    """
    Run an exploratory PC causal discovery baseline.

    alpha: is the treshold for rejecting the hypothesis of conditional independence.
    """
    ### 100 observations by 5 observed variables (economic_context, account_exposure, structuring_score, cross_border_ratio, transaction_volume).
    observations = data[features].to_numpy()
    ### PC helps identify what relationships exist betweem variables (using conditional independence) but not the direction.
    ### The statistical test "Fisher Z" is used: converts a Pearson correlation coefficient (r) into a normally distributed value (z).
    ### A graph's structural properties (d-separation) map directly to statistical independencies in the data.
    graph = pc(observations, alpha=alpha, indep_test=fisherz, show_progress=True)
    return graph


def discover_fci(data, features, alpha=0.05):
    """
    Run an exploratory FCI causal discovery baseline.
    """
    ### 100 observations by 5 observed variables (economic_context, account_exposure, structuring_score, cross_border_ratio, transaction_volume).
    observations = data[features].to_numpy()
    ### FCI is used to find hidden variables. It makes the supposition that there can be hidden variables that explain the relationship between the observed variables
    ### It returns "o" symbols which mean that FCI has not been able to completly understand the direction of the relationship
    ### Therefore, the existence of a hidden varaible is possible. 
    graph, edges = fci( observations, independence_test_method="fisherz", alpha=alpha, verbose=False, show_progress=False)
    return graph
