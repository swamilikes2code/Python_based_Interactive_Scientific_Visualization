# engine/check_datafile.py
#
# Shared CSV upload guards used by both Train and Test tabs, so a
# malformed or oversized upload fails fast with a clear message instead
# of propagating into pySINDy/solve_ivp and crashing with a low-level error.

import numpy as np

MAX_UPLOAD_BYTES = 5_000_000  # ~5MB of base64 text


def check_upload_size(b64_data):
    """Return an error string if the base64 payload is too large, else None."""
    if b64_data and len(b64_data) > MAX_UPLOAD_BYTES:
        return "File too large (max 5MB)."
    return None


def validate_dataframe(df):
    """Sanity-check a single loaded CSV DataFrame. Returns an error string or None."""
    if df.shape[1] < 2:
        return "CSV must have at least 2 columns: time + one state variable."
    if df.isnull().values.any():
        return "CSV contains missing/NaN values."
    if not np.isfinite(df.values).all():
        return "CSV contains infinite values."

    # CHANGED: require strictly increasing time. SmoothedFiniteDifference
    # and solve_ivp (used in engine.simulate) both assume t is monotonic —
    # a shuffled or duplicated-timestamp CSV would otherwise pass this
    # check and crash deep inside pysindy/scipy with a much less readable
    # error later on.
    t = df.iloc[:, 0].values
    if not np.all(np.diff(t) > 0):
        return ("Time column (first column) must be strictly increasing, "
                "with no duplicate timestamps.")
    return None


def validate_trajectory_set(dfs):
    """
    Sanity-check a LIST of DataFrames coming from a multi-file upload.
    All files must describe the SAME system measured from different
    initial conditions, so they must share an identical column layout
    (t, x1, x2, ...) in the same order — otherwise pooling them would
    silently mix unrelated state variables.

    Parameters
    ----------
    dfs : list[pd.DataFrame]

    Returns
    -------
    error string or None
    """
    if not dfs:
        return "No files uploaded."
    ref_cols = list(dfs[0].columns)
    for i, df in enumerate(dfs):
        if list(df.columns) != ref_cols:
            return (f"File #{i+1} columns {list(df.columns)} do not match the "
                    f"first file's columns {ref_cols}. All uploads must be "
                    "trajectories of the SAME system (identical columns).")
        if len(df) < 5:
            return f"File #{i+1} has too few rows ({len(df)})."
    return None