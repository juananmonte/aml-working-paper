
# Direct relationships between observable variables
TRUE_EDGES = [
    ("economic_context", "account_exposure"),
    ("economic_context", "transaction_volume"),
    ("account_exposure", "cross_border_ratio"),
    ("structuring_score", "transaction_volume"),
]


def evaluate_skeleton(discovered_edges):
    """Evaluate direct observable connections, ignoring direction."""

    # Ground truth connections
    truth = {
        frozenset(edge)
        for edge in TRUE_EDGES
    }

    # Discovered connections
    predicted = set()

    for edge in discovered_edges:

        # Example: "economic_context o-> transaction_volume"
        cause, symbol, effect = edge.split()

        predicted.add(frozenset((cause, effect)))

    # Compare discovered and true connections
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
