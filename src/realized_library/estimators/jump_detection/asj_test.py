from typing import Optional, Union
import numpy as np
from realized_library.utils.std_norm_dist_moments import mu_x
from realized_library.utils.resampling import compute as resample
from realized_library.estimators.variance.multipower_variation import compute as mpv

def compute(
    prices: np.ndarray,
    timestamps: np.ndarray,
    k: int,
    p: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    TODO: Add a description of the ASJ test and its purpose.
    """

    dt_ns = timestamps[1] - timestamps[0] # Sampling interval in nanoseconds
    delta = dt_ns / (24 * 60 * 60 * 1e9)  # Convert to fraction of day

    # returns = np.diff(np.log(prices))
    # B_p_delta = np.sum( np.abs(returns[1:] - returns[:-1]) ** p )
    B_p_delta = mpv(prices=prices, m=1, r=p)

    resampled_prices, _ = resample(prices=prices, timestamps=timestamps, sample_size=k)
    # resampled_returns = np.diff(np.log(resampled_prices))
    # B_p_kdelta = np.sum( np.abs(resampled_returns[1:] - resampled_returns[:-1]) ** p )
    B_p_kdelta = mpv(prices=resampled_prices, m=1, r=p)

    S = B_p_kdelta / B_p_delta
    # A = ( delta**(1 - p * 0.5) / mu_x(p) )

    
