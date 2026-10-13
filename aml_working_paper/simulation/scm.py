
# True causal relationships in our synthetic DGP

TRUE_EDGES = [
    ("economic_context", "network_exposure"),
    ("network_exposure", "aml_state"),
    ("economic_context", "transaction_volume"),
    ("aml_state", "transaction_volume"),
    ("network_exposure", "cross_border_ratio"),
    ("aml_state", "cross_border_ratio"),
    ("aml_state", "structuring_score"),
    ("structuring_score", "transaction_volume"),
]

# Dependencies between consecutive time periods
TEMPORAL_EDGES = [
    ("economic_context", "economic_context"),
    ("aml_state", "aml_state"),
]


def get_true_edges():
    return set(TRUE_EDGES)


def get_temporal_edges():
    return set(TEMPORAL_EDGES)
