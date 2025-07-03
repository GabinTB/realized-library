from typing import Optional
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import gamma
from realized_library.estimators.realized_variance import compute as rv

C1: float = 2.0
C2: float = 2.3

def compute(
    prices: np.array,
    m: int = 2,     # Bipower variation by default
    r: int = 2,     # Default ri for bipower variation
    M: Optional[int] = None,
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

    n = len(prices)
    K = int(C1 * n**(0.5) )                              # Lag lenght in ticks
    M = int(n**(0.5) / (C1 * C2)) if M is None else M    # Number of blocks
    rs = np.ones(m) * (r/m)

    log_prices = np.log(prices)
    blocks = np.array_split(log_prices, M) # Non-overlapping blocks of log prices

    # Mean K-lagged returns for each non-onverlapping block
    mklr = np.array([1 / ( (n / M) - K + 1 ) * np.sum(block[K:] - block[:-K]) for block in blocks if len(block) > K])
    if len(mklr) < 2:
        return 0.0
    
    mklr_windows = sliding_window_view(mklr, window_shape=m)     # Shape: (len(mklrs)-I+1, I)
    product_terms = np.prod(np.abs(mklr_windows) ** rs, axis=1)  # Products of r-powers of absolute returns

    scaling = n ** ( r * 0.25 - 0.5 )
    biais_scaling = n / (n - m + 1)

    return biais_scaling * scaling * np.sum(product_terms)