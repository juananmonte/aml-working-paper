def get_ground_truth():

    ### Define the true causal relationships of the whole data generation process
    current_edges = [
        ("economic_context", "account_exposure"),
        ("economic_context", "transaction_volume"),
        ("account_exposure", "aml_state"),
        ("account_exposure", "cross_border_ratio"),
        ("aml_state", "structuring_score"),
        ("aml_state", "cross_border_ratio"),
        ("aml_state", "transaction_volume"),
        ("structuring_score", "transaction_volume")
    ]

    #observable_edges = [
    #("economic_context", "account_exposure"),
    #("economic_context", "transaction_volume"),
    #("account_exposure", "cross_border_ratio"),
    #("structuring_score", "transaction_volume")
    #]

    ### True relationships between consecutive periods. Example, Economic ocntext t-1 to Economic Context t
    temporal_edges = [("economic_context", "economic_context", 1), ("aml_state", "aml_state", 1)]

    return current_edges, temporal_edges #observable_edges, 
