from numba import njit
import numpy as np

@njit
def compute(
    returns: np.ndarray,
    theta: float = 0.5,
) -> np.ndarray:
    """
    Compute the pre-average of returns using a specified theta.

    Parameters
    ----------
    returns : np.ndarray
        An array of returns.

    Returns
    -------
    np.ndarray
        The pre-averaged returns.
    """
    if len(returns) < 2:
        raise ValueError("Input array must contain at least two prices.")

    n = len(returns)
    K = int(np.ceil(np.sqrt(n) * theta))
    if K < 2 or K > n:
        raise ValueError(f"Invalid pre-averaging window K={K}, check theta and m.")

    w = np.minimum(np.arange(1, K) / K, 1 - np.arange(1, K) / K)

    preav_returns = np.empty(n - K + 2, dtype=np.float64)
    for i in range(n - K + 2):
        preav_returns[i] = np.dot(w, returns[i:i + K - 1])

    return preav_returns