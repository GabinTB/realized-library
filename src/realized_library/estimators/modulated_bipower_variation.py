from typing import Optional
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import gamma
from realized_library.estimators.realized_variance import compute as rv

C1: float = 2.0
C2: float = 2.3

# def _mu_x(x: float) -> float:
#     """
#     Compute mu_x = E(|N(0,1)|^x).
#     """
#     return (2**(x / 2)) / np.sqrt(np.pi) * gamma((x + 1) / 2)

# def _mbv(
#     M: int,
#     B,
#     R: float,
#     prices
# ) -> float:
#     """
#     Computation of mbv_{t,M} for Modulated Bipower Variation (MBV).

#     Parameters
#     ----------
#     M : int
#         Total number of observations (len(prices)).
#     B : int
#         Number of blocks.
#     R : int
#         Lag length (number of ticks).
#     prices : np.ndarray
#         1D array of price levels.

#     Returns
#     -------
#     float
#         mbv_{t,M} core term.
#     """
#     log_prices = np.log(prices)
#     prices_blocks = np.array_split(log_prices, B)

#     mbvs = np.array([np.mean(block[R:] - block[:-R]) for block in prices_blocks if len(block) > R])
#     if len(mbvs) < 2:
#         return 0.0

#     return np.sum(np.abs(mbvs[:-1]) * np.abs(mbvs[1:]))

# def compute(
#     prices: np.array,
#     C1: float = 2,
#     C2: float = 2.3,
#     B = 6,
#     omega2_est: Optional[float] = None,
# ) -> float:
#     """
#     Computes the modulated bipower variation (MBV) for a given list of prices.

#     Parameters
#     ----------
#     prices : np.array
#         Array of prices for which to compute the modulated bipower variation.
#     B : int, optional
#         TODO: Description of B parameter. Default is 6.
#     omega2_est : Optional[float], optional
#         Estimated noise variance. If not provided, it will be computed from the prices.

#     Returns
#     -------
#     float
#         The computed modulated bipower variation.
#     """
#     if len(prices) < 2:
#         raise ValueError("At least two prices are required to compute modulated bipower variation.")

#     M = len(prices)
#     R = int(C1 * np.sqrt(M))
#     mu1 = _mu_x(1)
#     v1 = C1 * (3 * C2 - 4 + max((2 - C2)**3, 0)) / (3 * (C2 - 1)**2)
#     v2 = 2 * min(C2 - 1, 1) / (C1 * (C2 - 1)**2)

#     omega2_est = omega2_est if omega2_est is not None else rv(prices) / (2*M) # Rough estimate of noise variance

#     return ( ((C1 * C2) / (mu1**2)) * _mbv(M=M, B=B, R=R, prices=prices) - v2 * omega2_est ) / v1

def compute(
    prices: np.array,
    I: int = 2,       # Bipower variation by default
    ri: float = 1,    # Default ri for bipower variation
    M: Optional[int] = None,
) -> float:
    """
    Computes the modulated bipower variation (MBV) for a given list of prices.

    Parameters
    ----------
    prices : np.array
        Array of prices for which to compute the modulated bipower variation.
    I : int, optional
        The number of absolute returns to consider in the product term. Default is 2 (bipower variation).
    ri : float, optional
        The exponent for the absolute returns in the product term. Default is 1 (bipower variation).
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
    K = int(C1 * np.sqrt(n) )                            # Lag lenght in ticks
    M = int(n**(1/2) / (C1 * C2)) if M is None else M    # Number of blocks
    r = np.ones(I) * ri

    log_prices = np.log(prices)
    blocks = np.array_split(log_prices, M) # Non-overlapping blocks of log prices

    # Mean K-lagged returns for each non-onverlapping block
    mklr = np.array([np.mean(block[K:] - block[:-K]) for block in blocks if len(block) > K])
    if len(mklr) < 2:
        return 0.0
    
    mklr_windows = sliding_window_view(mklr, window_shape=I)    # Shape: (len(mklrs)-I+1, I)
    product_terms = np.prod(np.abs(mklr_windows) ** r, axis=1)  # Products of r-powers of absolute returns

    scaling = n ** ( np.sum(r) * 0.25 - 0.5 ) # n ** ( np.sum(r) / 4 - 0.5 )
    bias_scale = n / (n - I)

    return bias_scale * scaling * np.sum(product_terms)