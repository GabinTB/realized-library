import warnings
from typing import Optional
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from realized_library.estimators.variance.bipower_variation import compute as bpv
from realized_library.estimators.variance.multipower_variation import compute as mpv
from realized_library._utils.std_norm_dist_moments import mu_x

def _treshold(
    alpha: float = 0.01,
) -> float:
    return -np.log(-np.log(1 - alpha))

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
    return value > _treshold(alpha=alpha)

def compute(
    prices: np.ndarray,
    K: int, #Optional[int] = 270,
    prev_day_prices: Optional[np.ndarray] = None, # When K is large, then prev_day_prices could be provided to fill missing data
    trading_day: int = 252,
) -> float:
    """
    Compute the Lee and Mykland Jump Test flags for a given series of prices. This test has custom thresholds 
    provided by the authors. This test returns intraday level jump flags or statistics, which differs from the
    other jump tests in this library that return only statistics.
    "Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics"
        By Lee, S.S., and Mykland P.A.
        DOI: 10.1093/rfs/hhm056
    
    Parameters
    ----------
    prices : np.ndarray
        1D array of prices for the day or 2D array of daily prices with shape (m, n) (m days, n data points per day).
    K : int
        The number of observations to consider for the jump test. It should be a positive integer.
    prev_day_prices : Optional[np.ndarray]
        Optional. If provided, it should be a 1D array of prices from the previous day. It will be concatenated with the current day's prices to form the final series.
    trading_day : int
        The number of trading days in a year. Default is 252, which is common for many stock markets.
    alpha : float
        The significance level for the test. Default is 0.01, which corresponds to a 99% confidence level.

    Returns
    -------
    np.ndarray
        A boolean array indicating the presence of jumps in the price series. True indicates a jump,
        and False indicates no jump.
    
    Raises
    ------
    ValueError
        If the input prices are empty, if K is not a positive integer, if prev_day_prices is provided but does not contain enough entries, or if the timestamps are not equally spaced.
    Warning
        If the computed K is outside the suggested bounds based on the number of observations.
    """
    # if prices.ndim > 1:
    #     statistics = []
    #     for price_series in prices:
    #         if len(price_series) < 2:
    #             raise ValueError("Each daily series must contain at least two entries.")
    #         statistics.append(compute(price_series, K, ...
    # TODO: Handle multiple days
    
    if prev_day_prices is not None:
        if len(prev_day_prices) < K:
            raise ValueError(f"If prev_day_prices provided, it must contain at least {K} entries to be relevant, but got {len(prev_day_prices)}.")
        prev_day_prices = prev_day_prices[-K:]
        final_prices = np.concatenate((prev_day_prices, prices))
    else:
        final_prices = prices
    
    n = len(prices)
    lb = np.sqrt(trading_day * n)
    hb = trading_day * n
    # if K is None:
    #     K = min( n // 2, int((lb + hb) * 0.5) )
    if not lb <= K <= hb:
        warnings.warn(f"Lee and Mykland Test suggests {lb} < K < {hb} but you chose K = {K} (for {trading_day} tradng days).")

    subsamples = sliding_window_view(final_prices, window_shape=K)
    if prev_day_prices is not None:
        subsamples = subsamples[-n:] # We should have len(prev_day_prices) == K
    
    Li = np.array([0.0] + [ np.log(subsamples[i][-1] / subsamples[i-1][-1]) / bpv(subsamples[i-1]) for i in range(1, len(subsamples)) ])
    # Li = np.array([0.0] + [ np.log(subsamples[i][-1] / subsamples[i-1][-1]) / mpv(subsamples[i-1]) for i in range(1, len(subsamples)) ])

    n -= 1 # Adjust for the first element being not defined
    c = mu_x(1)
    Sn = 1 / (c * (2 * np.log(n))**0.5)
    Cn = ((2 * np.log(n))**0.5) / c - 0.5 * (np.log(np.pi) + np.log(np.log(n))) * Sn
    
    return (np.max(np.abs(Li)) - Cn) / Sn

def flags(
    prices: np.ndarray,
    K: int,
    prev_day_prices: Optional[np.ndarray] = None, # When K is large, then prev_day_prices could be provided to fill missing data
    trading_day: int = 252,
    alpha: float = 0.01,
) -> np.ndarray:
    """
    Compute the Lee and Mykland Jump Test flags for a given series of prices. This test has custom thresholds 
    provided by the authors. This function returns intraday jump flags (LM test differs from the other jump tests 
    in this library that return only statistics).
    "Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics"
        By Lee, S.S., and Mykland P.A.
        DOI: 10.1093/rfs/hhm056
    
    Parameters
    ----------
    prices : np.ndarray
        1D array of prices for the day or 2D array of daily prices with shape (m, n) (m days, n data points per day).
    K : int
        The number of observations to consider for the jump test. It should be a positive integer.
    prev_day_prices : Optional[np.ndarray]
        Optional. If provided, it should be a 1D array of prices from the previous day. It will be concatenated with the current day's prices to form the final series.
    trading_day : int
        The number of trading days in a year. Default is 252, which is common for many stock markets.
    alpha : float
        The significance level for the test. Default is 0.01, which corresponds to a 99% confidence level.

    Returns
    -------
    np.ndarray
        A boolean array indicating the presence of jumps in the price series. True indicates a jump,
        and False indicates no jump.
    
    Raises
    ------
    ValueError
        If the input prices are empty, if K is not a positive integer, if prev_day_prices is provided but does not contain enough entries, or if the timestamps are not equally spaced.
    Warning
        If the computed K is outside the suggested bounds based on the number of observations.
    """
    
    if prev_day_prices is not None:
        if len(prev_day_prices) < K:
            raise ValueError(f"If prev_day_prices provided, it must contain at least {K} entries to be relevant, but got {len(prev_day_prices)}.")
        prev_day_prices = prev_day_prices[-K:]
        final_prices = np.concatenate((prev_day_prices, prices))
    else:
        final_prices = prices

    n = len(prices)
    lb = np.sqrt(trading_day * n)
    hb = trading_day * n
    if not lb <= K <= hb:
        warnings.warn(f"Lee and Mykland Test suggests {lb} < K < {hb} but you chose K = {K}.")

    subsamples = sliding_window_view(final_prices, window_shape=K)
    if prev_day_prices is not None:
        subsamples = subsamples[-n:] # We should have len(prev_day_prices) == K
        start_idx_jump_flags = 0
    else:
        start_idx_jump_flags = K

    Li = np.array([0.0] + [np.log(subsamples[i][-1] / subsamples[i-1][-1]) / bpv(subsamples[i-1]) for i in range(1, len(subsamples))])

    c = mu_x(1)
    Sn = 1 / (c * (2 * np.log(n))**0.5)
    Cn = ((2 * np.log(n))**0.5) / c - 0.5 * (np.log(np.pi) + np.log(np.log(n))) * Sn
    statistics = (np.abs(Li) - Cn) / Sn

    threshold = _treshold(alpha=alpha)

    jump_flags = np.zeros(n, dtype=bool)
    jump_flags[start_idx_jump_flags:] = statistics > threshold # Jump Flags: True if there is jump, False otherwise

    return jump_flags