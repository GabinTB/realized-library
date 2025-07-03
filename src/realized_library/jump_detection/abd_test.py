from typing import Union
import numpy as np
from realized_library._utils.std_norm_dist_moments import mu_x
from realized_library.estimators.realized_variance import compute as rv
from realized_library.estimators.bipower_variation import compute as bpv
from realized_library.estimators.multipower_variation import compute as mpv


def compute(
    prices: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Compute the BNS (Barndorff-Nielsen and Shephard) Jump Test statistic for one day.

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
    if prices.ndim > 1:
        statistics = []
        for price_series in prices:
            if len(price_series) < 2:
                raise ValueError("Each daily series must contain at least two entries.")
            statistics.append(compute(price_series))
        return np.array(statistics)
    
    returns = np.diff(np.log(prices))
    n = len(returns)
    if n < 4:
        raise ValueError("Need at least 4 observations for the BNS jump test.")

    mu1 = mu_x(1)
    RV = rv(prices)
    # BPV = (np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))) / (mu1**2)
    BPV = mpv(prices, 2, 2) # = bpv(prices)
    # QV = np.sum(np.abs(returns[3:]) * np.abs(returns[2:-1]) * np.abs(returns[1:-2]) * np.abs(returns[:-3]))
    QV = mpv(prices, 4, 2)

    jump_stat = ( (1/n)**(-0.5) ) * ( (RV - BPV)/RV ) * ( ( mu1**(-4) + 2 * mu1**(-2) - 5 ) * max(1, QV * BPV**(-2) ) )**(-0.5)

    return jump_stat