
# Ground truth causal relationships from the synthetic DGP

def get_ground_truth():

    # Relationships within the same time period
    current_edges = [
        ("economic_context", "network_exposure"),
        ("economic_context", "transaction_volume"),
        ("network_exposure", "aml_state"),
        ("network_exposure", "cross_border_ratio"),
        ("aml_state", "structuring_score"),
        ("aml_state", "cross_border_ratio"),
        ("aml_state", "transaction_volume"),
        ("structuring_score", "transaction_volume"),
    ]

    # Relationships between consecutive periods
    temporal_edges = [
        ("economic_context", "economic_context", 1),
        ("aml_state", "aml_state", 1),
    ]

    return current_edges, temporal_edges
