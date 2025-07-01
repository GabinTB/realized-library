import warnings
from typing import Optional
import numpy as np
from bisect import bisect_left

def compute(
    prices: np.ndarray,
    timestamps: Optional[np.ndarray] = None, 
    subsample_time_min: Optional[str] = None,
    subsample_size: Optional[int] = None,
) -> float:
    """
    Compute the realized Minimum Realized Variance (MinRV) from price data.

    Parameters
    ----------
    prices : np.ndarray
        Array of asset prices.
    timestamps : np.ndarray
        Array of timestamps corresponding to the prices.
    resampling_freq : str, optional
        Frequency for resampling the data (e.g., '1min', '5min'). If None, no resampling is performed.
    resampling_size : int, optional
        Size of the resampling window. If None, defaults to 1.

    Returns
    -------
    float
        The computed realized variance.
    """
    n = len(prices)
    if n < 2:
        raise ValueError("At least two prices are required to compute the realized variance.")
    
    if subsample_time_min is not None and timestamps is not None:
        warnings.warn("Both subsample_time_min and timestamps are provided. Resampling will be performed based on the frequency.")
    
    if subsample_time_min is not None:
        if timestamps is None:
            raise ValueError("Timestamps must be provided when subsample_time_min is specified.")
        total_time_ns = timestamps[-1] - timestamps[0]
        subsample_time_ns = int(subsample_time_min) * 60 * 1e9
        nb_subsamples = int(total_time_ns // subsample_time_ns) + 1
        minRVs = []
        for i in range(nb_subsamples):
            ts_start = timestamps[0] + i * subsample_time_ns
            ts_end = min(ts_start + subsample_time_ns, timestamps[-1])
            idx_start = bisect_left(timestamps, ts_start)
            idx_end = bisect_left(timestamps, ts_end)
            resampled_prices = prices[idx_start:idx_end]
            if len(resampled_prices) < 2:
                continue
            minRVs.append(compute(resampled_prices, None, None, None))
        return np.mean(minRVs)

    elif subsample_size is not None:
        minRVs = []
        for i in range(0, n, subsample_size):
            resampled_prices = prices[i:min(i + subsample_size, n)]
            if len(resampled_prices) < 2:
                continue
            minRVs.append(compute(resampled_prices, None, None, None))
        return np.mean(minRVs)
    
    else:
        m = len(prices)
        if m < 2:
            raise ValueError("At least two prices are required to compute the minRV.")
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        # returns2 = (np.sqrt(m) * returns) ** 2
        returns2 = (returns ** 2)
        r2_matrix = np.column_stack([returns2[:-1], returns2[1:]])  # Rolling window: each row has returns^2 at t, t+1
        min_rv_scale = np.pi / (np.pi - 2)
        return min_rv_scale * (m / (m - 1)) * np.sum(np.min(r2_matrix, axis=1))