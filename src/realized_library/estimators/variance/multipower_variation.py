from typing import Optional, Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pandas import to_timedelta
from realized_library._utils.std_norm_dist_moments import mu_x
from realized_library.utils.subsampling import compute as subsample

def compute(
    prices: list[float],
    m: int = 3,       # Tripower variation by default
    r: int = 2,       # Default tripower variation
    timestamps: Optional[np.array] = None,
    sample_size: Optional[Union[int, str]] = None,
    offset: Optional[Union[int, str]] = None
) -> float:
    """
    Computes multipower variation (MPV) for a given list of prices.
    "Jump-robust volatility estimation using nearest neighbor truncation"
        by Andersen, T. G., Dobrev, D., and Schaumburg, E. (2012).
        DOI: 10.1016/j.jeconom.2012.01.011

    Examples of multipower variation include:
    - Power Variation = MVP(m=1, r=2) = Realized Variance
    - Bipower Variation = MVP(m=2, r=2)
    - Tripower Variation = MVP(m=3, r=2)
    - Quadpower Variation = MVP(m=4, r=2)
    - Power Quarticity = MVP(m=1, r=4)
    - Bipower Quarticity = MVP(m=2, r=4)
    - Tripower Quarticity = MVP(m=3, r=4)
    - Quadpower Quarticity = MVP(m=4, r=4)

    Parameters
    ----------
    prices : list[float]
        List of prices for which to compute the multipower variation.
    m : int
        The number of powers to use in the multipower variation. Default is 3 for tripower variation.
    r : int
        The power to which the absolute returns are raised. Default is 2 for tripower variation
    timestamps : Optional[np.array], optional
        Timestamps corresponding to the prices, used for subsampling. If provided, must match the length of prices.
    sample_size : Optional[Union[int, str]], optional
        The size of the sample to be used for subsampling. If provided, must be a multiple of offset.
    offset : Optional[Union[int, str]], optional
        The offset for subsampling. If provided, must be a multiple of sample_size.

    Returns
    -------
    float
        The computed multipower variation.
    """
    if len(prices) < 2:
        raise ValueError("At least two prices are required to compute bipower variation.")
    
    rs = np.ones(m) * (r/m)

    if sample_size is not None and offset is not None:
        if timestamps is None:
            raise ValueError("Timestamps must be provided when using sample_size and offset parameters.")
        if isinstance(sample_size, str) and isinstance(offset, str):
            sample_size_ns = int(to_timedelta(sample_size).total_seconds() * 1e9)
            offset_ns = int(to_timedelta(offset).total_seconds() * 1e9)
            if sample_size_ns % offset_ns != 0:
                raise ValueError(f"Sample size {sample_size} must be a multiple of offset {offset} to reduce computation time.")
            nb_samples = sample_size_ns // offset_ns
        elif isinstance(sample_size, int) and isinstance(offset, int):
            if sample_size % offset != 0:
                raise ValueError(f"Sample size {sample_size} must be a multiple of offset {offset} to reduce computation time.")
            nb_samples = sample_size // offset
        else:
            raise ValueError("Both sample_size and offset must be either strings or integers.")

        price_subsamples, timestamps_subsamples = subsample(
            prices=prices, 
            timestamps=timestamps, 
            sample_size=sample_size, 
            offset=offset,
            nb_samples=nb_samples
        )

        N = len(prices)
        mvs = np.zeros(len(price_subsamples))
        total_count = 0
        for idx, sample in enumerate(price_subsamples):
            if len(sample) < m + 1:
                mvs[idx] = np.nan
            else:
                returns = np.diff(np.log(sample))
                n = len(returns)
                mklr_windows = sliding_window_view(returns, window_shape=m)
                product_terms = np.prod(np.abs(mklr_windows) ** rs, axis=1)  # Products of r-powers of absolute returns
                if idx == 0:
                    base_count = n - 1
                total_count += n - 1
                dmr = mu_x(r/m)**(-m)
                scaling = n ** ( r * 0.5 - 1.0 )
                mvs[idx] = dmr * scaling * np.sum(product_terms)

        mvSS = np.sum(mvs) * (base_count / total_count)
        biais_scaling = N / (N - m + 1)
        return biais_scaling * mvSS

    returns = np.diff(np.log(prices))
    n = len(returns)
    
    mklr_windows = sliding_window_view(returns, window_shape=m)
    product_terms = np.prod(np.abs(mklr_windows) ** rs, axis=1)  # Products of r-powers of absolute returns

    # np.sum(rs) == r, so we use r directly
    dmr = mu_x(r/m)**(-m)
    scaling = n ** ( r * 0.5 - 1.0 )
    biais_scaling = n / (n - m + 1)

    return  biais_scaling * scaling * dmr * np.sum(product_terms)