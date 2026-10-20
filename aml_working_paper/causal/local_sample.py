
import pandas as pd


CAUSAL_VARIABLES = [
    "economic_context",
    "network_exposure",
    "structuring_score",
    "cross_border_ratio",
    "transaction_volume",
]


def prepare_local_samples(df, selections, window_size=30 ):

    samples = []

    for strategy, windows in selections.items():

        for _, window in windows.iterrows():

            account_id = window["account_id"]
            start = window["start_time"]
            end = window["end_time"]

            # Get all observations for this account
            account_df = df[df["account_id"] == account_id].sort_values("time")

            # Keep observations up to the selected window's end
            history = account_df[account_df["time"] <= end]

            # Select the last 30 observations
            local_df = history.tail(window_size).copy()

            # Skip windows without enough history
            if len(local_df) != window_size:
                continue

            # Ensure the selected routing window is included
            start_end_filter = (account_df["time"] >= start) & (account_df["time"] <= end)
            selected_times = account_df[start_end_filter]["time"]

            if len(selected_times) != end - start + 1:
                continue

            # Keep only observable causal variables
            causal_data = local_df[CAUSAL_VARIABLES].copy()

            samples.append({
                "investigation_id": f"{strategy}_{len(samples):03d}",
                "strategy": strategy,
                "account_id": account_id,
                "routing_start": start,
                "routing_end": end,
                "causal_start": local_df["time"].iloc[0],
                "causal_end": local_df["time"].iloc[-1],
                "data": causal_data,
            })

    return samples
