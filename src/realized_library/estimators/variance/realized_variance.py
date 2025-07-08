import numpy as np

def compute(
    prices: np.ndarray
) -> float:
    """
    Compute the realized variance (sum of squared log returns) from high-frequency prices.

    Parameters
    ----------
    prices : np.ndarray
        Array of strictly positive price observations.

    Returns
    -------
    float
        Realized variance of the (resampled) price series.

    Raises
    ------
    ValueError
        For invalid inputs.
    """
    if np.any(prices <= 0):
        raise ValueError("Prices must be strictly positive for log-return calculation.")

    log_prices = np.log(prices)
    log_returns = np.diff(log_prices)
    return np.sum(log_returns ** 2)
