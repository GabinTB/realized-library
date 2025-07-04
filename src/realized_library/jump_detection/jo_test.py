from typing import Literal, Optional
import numpy as np
from realized_library.estimators.realized_variance import compute as rv
from realized_library.estimators.bipower_variation import compute as bpv
from realized_library.estimators.multipower_variation import compute as mpv


def compute(
    prices: np.ndarray,
    test: Literal["difference", "logarithmic", "ratio"] = "ratio",
    iv_est: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the Lee and Mykland Jump Test flags for a given series of prices.
    "Testing for Jumps When Asset Prices are Observed with Noise - A Swap Variance Approach"
        By Jiang G.J., and Oomen R.C.A. (2008).
        DOI: 10.1016/j.jeconom.2008.04.009
    
    Parameters
    ----------
    prices : np.ndarray
        1D array of prices for the day or 2D array of daily prices with shape (m, n) (m days, n data points per day).
    ...

    Returns
    -------
    ...

    Raises
    ------
    ...
    """

    simple_returns = prices[1:] / prices[:-1] - 1 # Ri
    log_returns = np.diff(np.log(prices)) # ri
    SwV = 2 * np.sum(simple_returns - log_returns)
    RV = rv(prices)
    BPV = bpv(prices) # mpv(prices, 2, 2)
    OMEGA = mpv(prices, 4, 2)

    if test == "difference":
        pass
    elif test == "logarithmic":
        pass
    elif test == "ratio":
        pass
    else:
        raise ValueError("Test must be one of 'difference', 'logarithmic', or 'ratio'.")
