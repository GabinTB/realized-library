from typing import Literal
import numpy as np
from scipy.stats import norm
from realized_library._utils.hft_timeseries_data import get_time_delta
from realized_library._utils.std_norm_dist_moments import mu_x
from realized_library.estimators.variance.realized_variance import compute as rv
from realized_library.estimators.variance.bipower_variation import compute as bpv
from realized_library.estimators.variance.multipower_variation import compute as mpv

def is_jump(
    value: float,
    alpha: float = 0.01,
) -> float:
    """
    Check if the value exceeds the threshold i.e. if the null hypothesis of no jump can be rejected,
    based on the significance level alpha.
    
    Parameters
    ----------
    value : float
        The computed jump test statistic.
    alpha : float
        The significance level for the test. Default is 0.01, which corresponds to a 99% confidence level.

    Returns
    -------
    bool
        True if the value indicates a jump, False otherwise.
    """
    return abs(value) > norm.ppf(1 - alpha)

def compute(
    prices: np.ndarray,
    timestamps: np.ndarray,
    test: Literal["linear", "ratio", "adjusted-ratio"] = "adjusted-ratio",
) -> float:
    """
    Compute the BNS (Barndorff-Nielsen and Shephard) Jump Test statistic for one day.
    "Econometrics of Testing for Jumps in Financial Economics Using Bipower Variation"
        by Barndorff-Nielsen, O.E., and Shephard, N. (2006).
        DOI: 10.1093/jjfinec/nbi022

    Parameters
    ----------
    prices : np.ndarray
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
    if len(timestamps) < 2:
        raise ValueError("Timestamps must contain at least two entries.")
    if np.diff(timestamps, n=2).any():
        raise ValueError("Timestamps must be equally spaced. Please resample the data before applying the test.")
    
    if prices.ndim > 1:
        statistics = []
        for price_series, timestamp_series in zip(prices, timestamps):
            if len(price_series) < 2 or len(timestamp_series) < 2:
                raise ValueError("Each daily series must contain at least two entries.")
            statistics.append(compute(prices=price_series, timestamps=timestamp_series))
        return np.array(statistics)

    n = len(prices) - 1
    if n < 4:
        raise ValueError("Need at least 4 observations for the BNS jump test.")
    
    t = (timestamps[-1] - timestamps[0]) / (24 * 60 * 60 * 1e9 - 1) # 1 day in nanoseconds - 1 ns to exclude the end of day timestamp
    delta = get_time_delta(timestamps=timestamps)
    returns = np.diff(np.log(prices))
    mu1 = mu_x(1)

    W = ((np.pi**2) / 4) + np.pi - 5 # ≈ 0.6090
    RV = rv(prices)
    BPV = np.sum(np.abs(returns[1:]) * np.abs(returns[:-1])) # = simplified bpv(prices) = mpv(prices, 2, 2)
    QPQ = (delta**(-1)) * np.sum(np.abs(returns[3:]) * np.abs(returns[2:-1]) * np.abs(returns[1:-2]) * np.abs(returns[:-3])) # = simplified mpv(prices, 4, 4) 

    if test == "linear": # G^
        return (delta**(-0.5)) * ( mu1**(-2) * BPV - RV ) / np.sqrt( W * mu1**(-4) * QPQ )
    elif test == "ratio": # H^
        return (delta**(-0.5)) * ( (mu1**(-2) * BPV / RV) - 1 ) / np.sqrt( W * ((QPQ / BPV)**2) ) 
    elif test == "adjusted-ratio": # J^
        return (delta**(-0.5)) * ( (mu1**(-2) * BPV / RV) - 1 ) / np.sqrt( W * max(t**(-1), QPQ / (BPV**2)) )