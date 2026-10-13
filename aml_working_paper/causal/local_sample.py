import pandas as pd


CAUSAL_VARIABLES = ["economic_context", "network_exposure", "structuring_score", "cross_border_ratio", "transaction_volume"]

def prepare_local_samples(df, selections):
    samples = []
    for strategy, windows in selections.items():
        ### itertuples is faster then iterrows.
        for window in windows.itertuples(index=False):
            account_id = window.account_id
            start = window.start_time
            end = window.end_time
            local_df = df[(df["account_id"] == account_id) & (df["time"] >= start) & (df["time"] <= end)]
            
            causal_data = local_df[CAUSAL_VARIABLES].sort_values("time")

            samples.append({"investigation_id": f"{strategy}_{len(samples):03d}", "strategy": strategy,
                            "account_id": account_id, "start_time": start, "end_time": end,
                            "data": causal_data
                            })
        return samples