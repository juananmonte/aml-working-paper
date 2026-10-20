
def prepare_local_windows(data, selected, features):
    """Extract original observations for selected investigation windows."""

    windows = []

    for _, window in selected.iterrows():

        # Get observations from the selected account
        account = data[data["account_id"] == window["account_id"]].sort_values("time").reset_index(drop=True)

        # Get the selected window positions
        start = int(window["start_idx"])
        end = int(window["end_idx"])

        # Extract observations and observable features
        local_data = account.iloc[start:end + 1][["time"] + features].copy()

        # Check that the correct observations were extracted
        if (len(local_data) != end - start + 1 or local_data["time"].iloc[0] != window["start_time"] or local_data["time"].iloc[-1] != window["end_time"]):
            raise ValueError("Window does not match original data")

        windows.append({
            "account_id": window["account_id"],
            "start_time": window["start_time"],
            "end_time": window["end_time"],
            "routing_score": window["routing_score"],
            "data": local_data,
        })

    return windows
