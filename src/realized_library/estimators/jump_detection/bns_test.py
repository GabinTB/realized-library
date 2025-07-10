from typing import Union, List
import numpy as np
from pandas import to_datetime, Timedelta
from realized_library._utils.hft_timeseries_data import get_time_delta
from realized_library._utils.std_norm_dist_moments import mu_x
from realized_library.estimators.variance.realized_variance import compute as rv
from realized_library.estimators.variance.bipower_variation import compute as bpv
from realized_library.estimators.variance.multipower_variation import compute as mpv


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
        for i, price_series, timestamp_series in zip(range(prices.shape[0]), prices, timestamps):
            if len(price_series) < 2 or len(timestamp_series) < 2:
                raise ValueError("Each daily series must contain at least two entries.")
            statistics.append(compute(prices=price_series, timestamps=timestamp_series))
        return np.array(statistics)

    n = len(prices) - 1
    if n < 4:
        raise ValueError("Need at least 4 observations for the BNS jump test.")
    
    start_ts = timestamps[0]
    start_of_day = to_datetime(start_ts, unit='ns').normalize()
    start_day_ts = int(start_of_day.timestamp() * 1e9)
    end_ts = timestamps[-1]
    end_of_day = start_of_day + Timedelta(days=1) # Exclude, so we'll remove 1 nanosecond at next line
    end_day_ts = int(end_of_day.timestamp() * 1e9) - 1 # ns
    
    t = (end_ts - start_day_ts) / (end_day_ts - start_day_ts)
    delta = get_time_delta(timestamps=timestamps)

    mu1 = mu_x(1)
    W = ((np.pi**2) / 4) + np.pi - 5 # ≈ 0.6090
    RV = rv(prices)
    BPV = mpv(prices, 2, 2) # = bpv(prices) = (np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))) / (mu1**2)
    QPQ = mpv(prices, 4, 4) # mpv(prices, 4, 2) #np.sum(np.abs(returns[3:]) * np.abs(returns[2:-1]) * np.abs(returns[1:-2]) * np.abs(returns[:-3]))

    # jump_stat = ( (mu1**(-2) * BPV / RV) - 1 ) * (delta**(-0.5)) / np.sqrt( W * max(1, QPQ / (BPV**2)) )
    jump_stat = ( (mu1**(-2) * BPV / RV) - 1 ) * (delta**(-0.5)) / np.sqrt( W * max(t**(-1), QPQ / (BPV**2)) )

    return jump_stat