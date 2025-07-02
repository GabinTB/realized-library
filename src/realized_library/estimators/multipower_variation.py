import warnings
from typing import Optional, Union
import numpy as np
from pandas import to_timedelta
from scipy.special import gamma
from realized_library.utils.subsampling import compute as subsample
from realized_library.utils.preaverage import compute as preaverage
from realized_library.estimators.noise_variance import compute as noise_variance_estimation


def _mu_x(x: float) -> float:
    """
    Compute mu_x = E(|N(0,1)|^x).
    """
    return (2**(x / 2)) / np.sqrt(np.pi) * gamma((x + 1) / 2)

def compute(
    prices: list[float],
    I: int = 3,         # Tripower variation by default
    ri: float = 2/3,    # Default ri for tripower variation
    timestamps: Optional[np.array] = None,
    sample_size: Optional[Union[int, str]] = None,
    offset: Optional[Union[int, str]] = None
) -> float:
    """
    Computes multipower variation (MPV) for a given list of prices.
    Examples of multipower variation include:
    - Tripower variation (I=3, ri=2/3)
    - Tripower quarticity (I=3, ri=4/3)
    - Quadpower quarticity (I=4, ri=1)

    Parameters
    ----------
    prices : list[float]
        List of prices for which to compute the multipower variation.
    I : int, optional
        The number of absolute returns to consider in the product term. Default is 3 (tripower variation).
    ri : float, optional
        The exponent for the absolute returns in the product term. Default is 2/3 (tripower variation).
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
    
    r = np.ones(I) * ri
    mu_product = np.prod([_mu_x(ri) for ri in r])

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

        m = len(prices)
        mvs = np.zeros(len(price_subsamples))
        total_count = 0
        for idx, sample in enumerate(price_subsamples):
            if len(sample) < 2:
                mvs[idx] = np.nan
            else:
                returns = np.diff(np.log(sample))
                n = len(returns)
                mpv_sum = 0.0
                for i in range(I, I + 1):
                    product_term = 1.0
                    for j in range(I):
                        product_term *= np.abs(returns[i - j - 1]) ** r[j]
                    mpv_sum += product_term
                if idx == 0:
                    base_count = n - 1
                total_count += n - 1
                
                scaling = n ** (np.sum(r) / 2 - 1)
                mvs[idx] = (1 / mu_product) * scaling * mpv_sum

        mvSS = np.sum(mvs) * (base_count / total_count)
        bias_scale = m / (m - I)
        return bias_scale * mvSS

    
    returns = np.diff(np.log(prices))
    n = len(returns)

    mpv_sum = 0.0
    for i in range(I, n + 1):
        product_term = 1.0
        for j in range(I):
            product_term *= np.abs(returns[i - j - 1]) ** r[j]
        mpv_sum += product_term

    scaling = n ** (np.sum(r) / 2 - 1)
    bias_scale = n / (n - I)
    return bias_scale * (1 / mu_product) * scaling * mpv_sum
