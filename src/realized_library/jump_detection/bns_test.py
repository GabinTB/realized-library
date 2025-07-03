from typing import Union, List
import numpy as np
from realized_library._utils.std_norm_dist_moments import mu_x
from realized_library.estimators.realized_variance import compute as rv
from realized_library.estimators.bipower_variation import compute as bpv
from realized_library.estimators.multipower_variation import compute as mpv


def compute(
    prices: np.ndarray,
    timestamps: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Compute the BNS (Barndorff-Nielsen and Shephard) Jump Test statistic for one day.
    "Econometrics of Testing for Jumps in Financial Economics Using Bipower Variation"
        by Barndorff-Nielsen, O.E., and Shephard, N. (2006).
        DOI: 10.1093/jjfinec/nbi022

    Parameters
    ----------
    intraday_returns : np.ndarray
        1D array of intraday data of shape (1,n) (n data points) for the day or 2D array of daily intraday 
        data with shape (m, n) (m days, n data points per day).
    timestamps : np.ndarray
        1D array of timestamps of shape (1,n) (n data points) corresponding to the intraday data, in nanoseconds 
        since epoch, or 2D array of daily timestamps with shape (m, n) (m days, n data points per day).

    Returns
    -------
    Union[float, np.ndarray]
        The BNS jump test statistic for the day or an array of statistics for multiple days.
    """
    if prices.shape != timestamps.shape:
        raise ValueError("Prices and timestamps must have the same shape.")
    
    if prices.ndim > 1:
        statistics = []
        for p, ts in zip(prices, timestamps):
            if len(p) < 2 or len(ts) < 2:
                raise ValueError("Each daily series must contain at least two entries.")
            statistics.append(compute(p, ts))
        return np.array(statistics)

    if len(timestamps) < 2:
        raise ValueError("Timestamps must contain at least two entries.")
    if np.diff(timestamps, n=2).any():
        raise ValueError("Timestamps must be equally spaced. Please resample the data before applying the test.")
    
    delta_ns = timestamps[1] - timestamps[0]  # Sampling interval in nanoseconds
    delta_sec = delta_ns / 1e9  # Convert to seconds
    d = delta_sec / (24 * 3600)  # Convert to fraction of day
    t = 1

    returns = np.diff(np.log(prices))
    n = len(returns)
    if n < 4:
        raise ValueError("Need at least 4 observations for the BNS jump test.")

    mu1 = mu_x(1)
    W = (np.pi**2 / 4) + np.pi - 5 # ≈ 0.6090
    RV = rv(prices)
    # BPV = (np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))) / (mu1**2)
    BPV = mpv(prices, 2, 2) # = bpv(prices)
    # QV = np.sum(np.abs(returns[3:]) * np.abs(returns[2:-1]) * np.abs(returns[1:-2]) * np.abs(returns[:-3]))
    QV = mpv(prices, 4, 2)

    jump_stat = ( (mu1**(-2) * BPV / RV) - 1 ) * d**(-0.5) / np.sqrt( W * max(t**(-1), QV * BPV**(-2)) )

    return jump_stat