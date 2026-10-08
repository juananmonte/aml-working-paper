# investigation_routing.py

import numpy as np
import pandas as pd


def build_candidate_windows(routing_df, window_size=10):
    """
    Create candidate investigation windows.

    Each window ends at time t and uses the routing
    signals available at that time.
    """

    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    candidates = []

    for account_id, account_df in routing_df.groupby("account_id"):

        account_df = account_df.sort_values("time").reset_index(drop=True)

        for end_idx in range(window_size - 1, len(account_df)):

            start_idx = end_idx - window_size + 1

            candidates.append({
                "account_id": account_id,
                "start_time": account_df.loc[start_idx, "time"],
                "end_time": account_df.loc[end_idx, "time"],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "uncertainty": account_df.loc[end_idx, "uncertainty"],
                "surprise": account_df.loc[end_idx, "surprise"],
            })

    return pd.DataFrame(candidates)

def select_investigations(
    candidates,
    budget=10,
    strategy="uncertainty",
    random_state=42,
):
    """
    Select investigation windows under a fixed budget.

    Strategies:
        uncertainty
        surprise
        combined
        random

    Windows from the same account cannot overlap.
    """

    if budget < 0:
        raise ValueError("budget must be >= 0")

    if strategy not in {
        "uncertainty", "surprise", "combined", "random"
    }:
        raise ValueError(f"Unknown strategy: {strategy}")

    ranked = candidates.copy()

    if ranked.empty or budget == 0:
        return ranked.iloc[:0].copy()

    if strategy == "uncertainty":
        ranked["routing_score"] = ranked["uncertainty"]

    elif strategy == "surprise":
        ranked["routing_score"] = ranked["surprise"]

    elif strategy == "combined":
        u_rank = ranked["uncertainty"].rank(pct=True)
        s_rank = ranked["surprise"].rank(pct=True)

        ranked["routing_score"] = 0.5 * (u_rank + s_rank)

    else:
        rng = np.random.default_rng(random_state)
        ranked["routing_score"] = rng.random(len(ranked))

    ranked = ranked.sort_values(
        ["routing_score", "account_id", "end_idx"],
        ascending=[False, True, True],
    )

    selected = []

    for _, candidate in ranked.iterrows():

        if len(selected) >= budget:
            break

        overlap = any(
            candidate["account_id"] == previous["account_id"]
            and candidate["start_idx"] <= previous["end_idx"]
            and candidate["end_idx"] >= previous["start_idx"]
            for previous in selected
        )

        if not overlap:
            selected.append(candidate)

    return pd.DataFrame(selected).reset_index(drop=True)

def validate_investigations(selected, candidates, budget):
    """
    Check budget, eligibility and non-overlapping windows.
    """

    assert len(selected) <= budget

    eligible = set(
        zip(
            candidates["account_id"],
            candidates["start_idx"],
            candidates["end_idx"],
        )
    )

    for _, window in selected.iterrows():
        assert (
            window["account_id"],
            window["start_idx"],
            window["end_idx"],
        ) in eligible

    for account_id, account_df in selected.groupby("account_id"):

        account_df = account_df.sort_values("start_idx")

        previous_end = None

        for _, window in account_df.iterrows():

            if previous_end is not None:
                assert window["start_idx"] > previous_end

            previous_end = window["end_idx"]

    print("Investigation selection validated.")