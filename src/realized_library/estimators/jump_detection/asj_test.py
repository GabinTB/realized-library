from typing import Optional, Union
import numpy as np
from scipy.special import gamma
from realized_library.utils.resampling import compute as resample
from realized_library.estimators.variance.trucated_multipower_variation import compute as tpv
from realized_library.estimators.variance.multipower_variation import compute as mpv

def _B_p(X: np.ndarray, p: int) -> float:
    delta_X = np.diff(np.log(X)) # Log-returns
    return np.sum(np.abs(delta_X) ** p)

def compute(
    prices: np.ndarray,
    timestamps: np.ndarray,
    k: int,
    p: Optional[int] = None,
    q: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Compute the Aït-Sahalia and Jacod (ASJ) Jump Test flags for a given series of prices.
    "Testing for Jumps in a Discretely Observed Process"
        By Aït-Sahalia Y., and Jacod J. (2009).
        DOI: 10.1214/07-AOS568

    Parameters
    ----------
    TODO: Add parameters description

    Returns
    ----------
    TODO: Add return description

    Raises
    ----------
    TODO: Add exceptions description
    """

    B_p_delta = _B_p(prices, p)

    resampled_prices, _ = resample(prices=prices, timestamps=timestamps, sample_size=k)
    B_p_kdelta = _B_p(resampled_prices, p)

    S = B_p_kdelta / B_p_delta

    # Realized truncated pth variation
    A = tpv(prices=prices, timestamps=timestamps, p=p, correct_scaling_bias=False)
    # Or Multipower Variation estimator: "The multipower variations (22) do not suffer from the drawback of having to choose α and omega a priori, but they cannot be used for Theorem 7"
    A = mpv(prices=prices, timestamps=timestamps, m=q, r=p, correct_noise=False)

    
