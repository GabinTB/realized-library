from typing import Optional
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from realized_library._utils.hft_timeseries_data import get_time_delta
from realized_library.estimators.variance.realized_variance import compute as rv

def compute(
    prices: np.array,
    omega2_est: float,
    timestamps: Optional[np.array] = None,
    m: int = 2,     # Bipower variation by default
    r: int = 2,     # Default ri for bipower variation
    M: Optional[int] = None,
    correct_scaling_bias: bool = True
) -> float:
    """
    Computes the modulated bipower variation (MBV) for a given list of prices.
    "Estimation of volatility functionals in the simultaneous presence of microstructure noise and jumps"
        by Podolskij, M., and Vetter, M. (2009).
        DOI: 10.3150/08-BEJ167

    Parameters
    ----------
    prices : np.array
        Array of prices for which to compute the modulated bipower variation.
    m : int
        The number of powers to use in the multipower variation. Default is 3 for tripower variation.
    r : int
        The power to which the absolute returns are raised. Default is 2 for tripower variation
    M : Optional[int], optional
        The number of blocks to split the prices into. If None, it will be computed based on the length of prices.

    Returns
    -------
    float
        The computed modulated bipower variation.
    """
    if len(prices) < 2:
        raise ValueError("At least two prices are required to compute modulated bipower variation.")

    n = len(prices) - 1
    biais_scaling = n / (n - m + 1) if correct_scaling_bias else 1.0

    c2 = 8/5
    c1 = np.sqrt( 18 / ((c2 - 1) * (4 - c2)) ) * np.sqrt(omega2_est) / rv(prices)  # c1 is a scaling constant based on omega2_est and realized variance
    K = int(c1 * (n**0.5))                              # Lag lenght in ticks
    M = int((n**0.5) / (c1 * c2)) if M is None else M   # Number of blocks
    rs = np.ones(m) * (r/m)
    
    log_prices = np.log(prices)
    blocks = np.array_split(log_prices, M) # Non-overlapping blocks of log prices

    # Mean K-lagged returns for each non-onverlapping block
    mklr = np.array([np.sum(block[K:] - block[:-K]) / ( (n / M) - K + 1 ) for block in blocks if len(block) > K])
    if len(mklr) < 2:
        return 0.0
    
    mklr_windows = sliding_window_view(mklr, window_shape=m)     # Shape: (len(mklrs)-I+1, I)
    product_terms = np.prod(np.abs(mklr_windows) ** rs, axis=1)  # Products of r-powers of absolute returns

    return  biais_scaling * (n ** (r * 0.25 - 0.5)) * np.sum(product_terms)