from typing import Union, Optional, Literal
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
    timestamps: Optional[np.ndarray] = None,
    test: Literal["difference", "ratio"] = "ratio",
    correct_for_noise: bool = True,
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
    if prices.ndim > 1:
        statistics = []
        for price_series, timestamp_series in zip(prices, timestamps):
            if len(price_series) < 2:
                raise ValueError("Each daily series must contain at least two entries.")
            statistics.append(compute(price_series, timestamp_series))
        return np.array(statistics)
    
    returns = np.diff(np.log(prices))
    n = len(returns)
    if n < 4:
        raise ValueError("Need at least 4 observations for the ABD test.")

    # t = (timestamps[-1] - timestamps[0]) / (24 * 60 * 60 * 1e9 - 1) # 1 day in nanoseconds - 1 ns to exclude the end of day timestamp
    delta = get_time_delta(timestamps=timestamps, N=n)
    mu1 = mu_x(1)

    if not correct_for_noise:
        RV = rv(prices)
        BV = bpv(prices) # = mpv(prices, 2, 2)
        TQ = mpv(prices, 3, 4)
    else:
        RV = rv(prices)
        BV = (mu1**(-2)) * ((1 - 2*delta)**(-1)) * np.sum(np.abs(returns[2:] * returns[:-2]))
        TQ = (delta**(-1)) * (mu_x(4/3)**(-3)) * ((1 - 4*delta)**(-1)) * np.sum(np.prod(np.power(np.abs(np.matrix([returns[4:], returns[2:-2], returns[:-4]]).T), 4/3), axis=1))
    
    if test == "difference":
        return ( delta**(-0.5) ) * ( RV - BV ) / ( ( (mu1**(-4) + 2 * mu1**(-2) - 5) * TQ )**0.5 )
    if test == "ratio":
        return ( delta**(-0.5) ) * ( (RV - BV) * RV**(-1) ) / ( ( (mu1**(-4) + 2 * mu1**(-2) - 5) * max(1, TQ * BV**(-2)) )**0.5 )
        # return ( delta**(-0.5) ) * ( (RV - BV) * RV**(-1) ) / ( ( (mu1**(-4) + 2 * mu1**(-2) - 5) * max(t**(-1), TQ * BV**(-2)) )**0.5 )
    else:
        raise ValueError("Test must be either 'difference' or 'ratio'.")

