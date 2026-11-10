
def prepare_local_windows(data, selected, features):
    """
    Extract original observations for selected investigation windows.
    """
    windows = []
    for _, window in selected.iterrows():
        ### Get original observations (data) from the selected window obtained by routing (selected)
        account = data[data["account_id"] == window["account_id"]].sort_values("time").reset_index(drop=True)
        ### Get the selected window start and end index
        start = int(window["start_idx"])
        end = int(window["end_idx"])
        ### Extract observations and the observable features that were defined
        local_data = account.iloc[start:end + 1][["time"] + features].copy()
        ### save them in a dictionary
        windows.append({
            "account_id": window["account_id"],
            "start_time": window["start_time"],
            "end_time": window["end_time"],
            "routing_score": window["routing_score"],
            "data": local_data,
        })

    return windows
