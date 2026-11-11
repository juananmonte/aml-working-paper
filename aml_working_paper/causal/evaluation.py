
from aml_working_paper.causal.ground_truth import get_ground_truth


def evaluate_skeleton(discovered_edges):
    """
    Compare and evaluate the found causal graphs.
    """
    ### Get the proper ground truth
    current_edges, _ = get_ground_truth()
    ### Observable variables used in causal discovery
    features = {"economic_context", "account_exposure","structuring_score", "cross_border_ratio","transaction_volume"}
    ### Keep only direct relationships between observed variables
    truth = {tuple(sorted((cause, effect)))for cause, effect in current_edges if cause in features and effect in features}
    ### Get discovered connections
    predicted = set()

    for edge in discovered_edges:
        cause, symbol, effect = edge.split()
        predicted.add(tuple(sorted((cause, effect))))

    # Compare connections
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0)

    metrics = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1 }

    return metrics

def summarize_causal_results(evaluation, random_results):
    """
    Summarize how the uncertainty behaved against random selections.
    With these, the investigation hypothesis can be anlyzed: window selection using HMM uncertainty gets better causal relationships thatn random slection
    """

    ### F1 obtained by uncertainty routing with PC
    uncertainty_f1 = evaluation.loc[(evaluation["method"] == "PC") & (evaluation["strategy"] == "uncertainty"), "f1"].iloc[0]
    ### Random baseline statistics
    random_f1 = random_results["f1"]

    random_better = 0

    for f1 in random_f1:
        if f1 >= uncertainty_f1:
            random_better += 1

    summary = {
        "uncertainty_f1": float(uncertainty_f1),
        "random_mean_f1": float(random_f1.mean()),
        "random_std_f1": float(random_f1.std()),
        "random_fraction_ge_uncertainty": random_better / len(random_f1),
        "random_repetitions": len(random_results)
    }
    return summary
