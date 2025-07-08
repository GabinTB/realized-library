from typing import Union
import numpy as np
from pandas import to_datetime, Timedelta
from realized_library._utils.std_norm_dist_moments import mu_x
from realized_library.estimators.variance.realized_variance import compute as rv
from realized_library.estimators.variance.bipower_variation import compute as bpv
from realized_library.estimators.variance.multipower_variation import compute as mpv


def compute(
    prices: np.ndarray,
    timestamps: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Compute the ABD (Andersen, Bollerslev, and Diebold) Jump Test statistic for one day.
    "Roughing it Up: Including Jump Components in the Measurement, Modeling and Forecasting of Return Volatility"
        by Andersen, T.G., Bollerslev, T., and Diebold, F.X. (2005).
        DOI: 10.3386/w11775

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
        The ABD jump test statistic for the day or an array of statistics for multiple days.
    """
    if prices.shape != timestamps.shape:
        raise ValueError("Prices and timestamps must have the same shape.")
    if np.diff(timestamps, n=2).any():
        raise ValueError("Timestamps must be equally spaced. Please resample the data before applying the test.")
    if prices.ndim > 1:
        statistics = []
        for price_series, timestamp_series in zip(prices, timestamps):
            if len(price_series) < 2:
                raise ValueError("Each daily series must contain at least two entries.")
            statistics.append(compute(price_series, timestamp_series))
        return np.array(statistics)
    
    n = len(prices) - 1
    if n < 4:
        raise ValueError("Need at least 4 observations for the ABD test.")

    start_ts = timestamps[0]
    start_of_day = to_datetime(start_ts, unit='ns').normalize()
    start_day_ts = int(start_of_day.timestamp() * 1e9)
    end_ts = timestamps[-1]
    end_of_day = start_of_day + Timedelta(days=1) # Exclude, so we'll remove 1 nanosecond at next line
    end_day_ts = int(end_of_day.timestamp() * 1e9) - 1 # ns
    t = (end_ts - start_day_ts) / (end_day_ts - start_day_ts)

    dt_ns = timestamps[1] - start_ts # Sampling interval in nanoseconds
    delta = dt_ns / (24 * 60 * 60 * 1e9)  # Convert to fraction of day

    mu1 = mu_x(1)
    RV = rv(prices)
    BPV = mpv(prices, 2, 2) # = bpv(prices)
    TPQ = mpv(prices, 3, 4)

    # jump_stat = ( delta**(-0.5) ) * ( max(0, RV - BPV) * RV**(-1) ) / ( (mu1**(-4) + 2 * mu1**(-2) - 5) * max(1, TPQ * BPV**(-2)) )**0.5
    jump_stat = ( delta**(-0.5) ) * ( max(0, RV - BPV) * RV**(-1) ) / ( (mu1**(-4) + 2 * mu1**(-2) - 5) * max(t**(-1), TPQ * BPV**(-2)) )**0.5

    return jump_stat