
import pandas as pd

def prepare_discovery_data(local_windows):
    """
    Make a dataframe from the selected windows obtained by routing
    """
    data = []
    for window_id, window in enumerate(local_windows):
        ### Get observations from the current window
        observations = window["data"].copy()
        ### Identify the window from where the data comes from and the corresponding account
        observations["window_id"] = window_id
        observations["account_id"] = int(window["account_id"])
        data.append(observations)
    ### Combine all windows into one DataFrame
    discovery_data = pd.concat(data,ignore_index=True,)
    return discovery_data
