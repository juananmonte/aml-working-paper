import numpy as np
import pandas as pd


def build_candidate_windows(routing_df, window_size=10):
    """
    Create one candidate for every sliding window of each account.
    """

    windows = []
    ### Split and order accounts to prevent mixing observations and also mantain the time order
    for account_id, account_data in routing_df.groupby("account_id"):
        account_data = account_data.sort_values("time").reset_index(drop=True)
        ### loop with sliding windows
        for end in range(window_size - 1, len(account_data)):
            start = end - window_size + 1

            ### For each window, obtain the LAST observation's metrics to check the models behaviour at the end of the window
            ### this will help with causal inference by having the previous observations' information
            windows.append({
                "account_id": account_id,
                "start_time": account_data.loc[start, "time"],
                "end_time": account_data.loc[end, "time"],
                "start_idx": start,
                "end_idx": end,
                "uncertainty": account_data.loc[end, "uncertainty"],
                "surprise": account_data.loc[end, "surprise"],
            })

    return pd.DataFrame(windows)


def select_investigations(candidates, budget=10, strategy="uncertainty", random_state=42):
    """
    Select the windows that were candidates for different accounts.
    The windows will be selected according to the the higher values for the selected strategy
    These will be used to find a AML representative causal structure for all the accounts.
    """

    ranked = candidates.copy()

    ###Select a strategy and assign a score to every candidate window
    if strategy == "uncertainty":
        ranked["routing_score"] = ranked["uncertainty"]
    elif strategy == "surprise":
        ranked["routing_score"] = ranked["surprise"]
    elif strategy == "combined":
        ### get the percentiles because surprice and uncertainty don't have the same scale. 
        uncertainty_rank = ranked["uncertainty"].rank(pct=True)
        surprise_rank = ranked["surprise"].rank(pct=True)
        ranked["routing_score"] = (uncertainty_rank + surprise_rank) / 2
    else:
        ### random routing score. This is used as the baselinefor comparison
        random_generator = np.random.default_rng(random_state)
        ranked["routing_score"] = random_generator.random(len(ranked))

    ### order the windows from the highest routing score to the lowest
    ranked = ranked.sort_values(["routing_score", "account_id", "end_idx"],ascending=[False, True, True])
    
    selected = []
    ### loop through the selected windows until the budget (max number of windows) is reached
    for _, window in ranked.iterrows():
        if len(selected) >= budget:
            break

        overlaps = False
        ### Avoid overlapping windows: not the same  obervations for the same period for the same account
        for previous in selected:
            same_account = window["account_id"] == previous["account_id"]
            same_period = ( window["start_idx"] <= previous["end_idx"] and window["end_idx"] >= previous["start_idx"])
            if same_account and same_period:
                overlaps = True
                break
        if not overlaps:
            selected.append(window)
            
    investigations = pd.DataFrame(selected).reset_index(drop=True)
    return investigations
