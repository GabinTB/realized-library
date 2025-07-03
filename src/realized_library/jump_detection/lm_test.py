import warnings
from typing import Optional
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from realized_library.utils.resampling import compute as resampling
from realized_library.utils.subsampling import compute as subsampling
from realized_library.estimators.bipower_variation import compute as bpv
from realized_library.estimators.multipower_variation import compute as mpv
from realized_library._utils.std_norm_dist_moments import mu_x

# # TODO: handling resampling
def compute(
    prices: np.ndarray,
    # timestamps: np.ndarray,
    K: Optional[int] = 270,
    past_day_prices: Optional[np.ndarray] = None, # When K is large, then past_day_prices could be provided to fill missing data
    # resampling_timeframe: Optional[str] = "5m",
    trading_day: int = 252,
    alpha: float = 0.01
) -> np.ndarray:
    """
    Compute...
    "Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics"
        By Lee, S.S., and Mykland P.A.
        DOI: 10.1093/rfs/hhm056
    """
    # if prices.shape != timestamps.shape:
    #     raise ValueError("Prices and timestamps must have the same shape.")
    # if len(timestamps) < 2:
    #     raise ValueError("Timestamps must contain at least two entries.")
    # if np.diff(timestamps, n=2).any():
    #     raise ValueError("Timestamps must be equally spaced. Please resample the data before applying the test.")
    
    if past_day_prices is not None:
        if len(past_day_prices) < K:
            raise ValueError(f"If past_day_prices provided, it must contain at least {K} entries to be relevant, but got {len(past_day_prices)}.")
        past_day_prices = past_day_prices[K:] # First K entries are not needed
        final_prices = np.concatenate((past_day_prices, prices))
    else:
        final_prices = prices
    
    # if resampling_timeframe is not None:
    #     prices, timestamps = resampling(prices=prices, timestamps=timestamps, sample_size=resampling_timeframe)

    n = len(prices)
    lb = np.sqrt(trading_day * n)
    hb = trading_day * n
    if K is None:
        K = min( n // 2, int((lb + hb) * 0.5) )
    if not lb <= K <= hb:
        warnings.warn(f"Lee and Mykland Test suggests {lb} < K < {hb} but you chose K = {K}.")

    if past_day_prices is not None:
        subsamples = sliding_window_view(np.log(final_prices), window_shape=K)[K:] # We should have len(past_day_prices) == K
    else:
        subsamples = sliding_window_view(np.log(final_prices), window_shape=K)
    
    Li = np.array([(sample[-1]/sample[-2]) / bpv(sample) for sample in subsamples if len(sample) >= 2])
    # Li = np.array([(sample[-1]/sample[-2]) / mpv(sample, 2, 2) for sample in subsamples])
    c = mu_x(1)
    Sn = (c*(2*np.log(n))**0.5)**(-1)
    Cn = (2 * np.log(n))**0.5 / c - 0.5 * (np.log(np.pi) + np.log(np.log(n))) * Sn
    threshold = -np.log(-np.log(1-alpha))
    
    return (np.abs(Li) - Cn) * Sn**(-1) > threshold # Jump Flags: True if there is jump, False otherwise