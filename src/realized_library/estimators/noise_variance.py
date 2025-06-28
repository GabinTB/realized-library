import numpy as np
from typing import Optional
from realized_library.estimators.realized_variance import compute as rv

def optimal_q(
    timestamps: np.ndarray,
    target_time_interval: int = 120,  # Minimum time interval in seconds (default is 2 minutes)
) -> int:
    """
    Estimate the optimal q parameter for robust ω^2 estimation.
    This function finds the largest q such that the average time interval between every q-th observation
    is closest to the target time interval (default is 2 minutes = 120 seconds).

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamps corresponding to the log returns. Must be nanoseconds timestamps.
    target_time_interval : int, optional
        Target average time interval in seconds for the estimation (default is 2 minutes = 120 seconds).

    Returns
    -------
    int
        Optimal q parameter for robust ω^2 estimation.
    """
    if len(str(int(timestamps[0]))) != 19:
        raise ValueError("Timestamps must be in nanoseconds (19 digits).")
    if not 1 <= target_time_interval <= 30 * 60:
        raise ValueError("target_time_interval should be between 1s and 30 minutes.")

    lowest_avg_interval_diff = np.inf
    for q in range(2, 10000): # 10000 is an arbitrary upper limit for q, experiments have shown that q rarely exceeds 5000 for very liquid assets
        
        time_diffs = []
        for i in range(q):
            subsampled_ts = timestamps[::q]
            time_diff_ns = subsampled_ts[1] - subsampled_ts[0]
            time_diff_sec = time_diff_ns / 1e9  # Convert nanoseconds to seconds
            time_diffs.append(time_diff_sec)
        avg_interval = np.mean(time_diffs)

        avg_interval_diff = abs(target_time_interval - avg_interval)  # Difference from 2 minutes
        if avg_interval_diff < lowest_avg_interval_diff:
            lowest_avg_interval_diff = avg_interval_diff
        else:
            if q - 1 <= 1:
                raise ValueError(
                    f"q must be greater than 1 for robust ω^2 estimation, but found q = {q - 1}."
                    "Consider increasing the target time interval relative to the event frequencies (timestamps)."
                )
            return q - 1 # q_opt is the last q that had a lower average time difference than the target

    raise ValueError(
        f"Could not find a suitable q for target time interval {target_time_interval} minutes. "
        "Consider adjusting the target time interval."
    )

def compute(
    prices: np.ndarray,
    q: Optional[float] = None,
) -> float:
    """
    Estimate the omega2 parameter from log returns.
    Estimation method from  https://doi.org/10.1111/j.1368-423X.2008.00275.x:
    computing the realised variance using every q-th trade or quote, leading to q distinct RVs,
    and then averaging these RVs to obtain omega2, with q > 1 for robustness.
    Ideally, choose q such that every q-th observation is, on average, 2 minutes apart, examples:
    - We recommend q = 50 for trade prices of liquid assets
    - We recommend q = 70 for quote prices of liquid assets
    - q = 120 for 1s close prices

    Parameters
    ----------
    prices : np.ndarray
        Array of strictly positive price observations.
    q : Optional[float], optional
        The q parameter for robust ω^2 estimation. If None, it will be estimated from

    Returns
    -------
    float
        Estimated omega2 parameter.
    """
    if q is not None and q < 2:
        raise ValueError("q must be greater to 1 for robust ω^2 estimation.")

    omega2_estimates = []
    for i in range(2, q+1):
        subsample_prices = prices[::q] # Subsample every q-th observation starting at offset i
        n = len(subsample_prices)
        
        if n < 2:
            continue  # Skip subsamples with too few points
        rv_dense_i = rv(subsample_prices)  # Compute realized variance for this subsample
        omega2_i = rv_dense_i / (2*n)
        omega2_estimates.append(omega2_i)
    
    if len(omega2_estimates) != q - 1:
        raise ValueError("We didn't obtain q distinct realised variances")

    return np.mean(omega2_estimates)