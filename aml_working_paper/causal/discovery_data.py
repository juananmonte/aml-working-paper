
import pandas as pd


def prepare_discovery_data(local_windows):
    """Combine selected windows while keeping their identities."""

    data = []

    for window_id, window in enumerate(local_windows):

        # Get observations from this window
        local_data = window["data"].copy()

        # Identify the window and account
        local_data["window_id"] = window_id
        local_data["account_id"] = int(window["account_id"])
        data.append(local_data)

    # Combine all windows into one DataFrame
    discovery_data = pd.concat(data,ignore_index=True,)

    return discovery_data
