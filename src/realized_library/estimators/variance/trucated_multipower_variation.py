from typing import Optional, Union
import numpy as np
from pandas import to_timedelta
from realized_library._utils.std_norm_dist_moments import mu_x
from realized_library._utils.hft_timeseries_data import get_time_delta
from realized_library.utils.subsampling import compute as subsample

def compute(
    prices: np.ndarray,
    p: int = 2, # Default is 2 for Realized Truncated Volatility
    alpha: Optional[float] = None,
    omega: float = 0.47,
    beta: float = 0.2,
    timestamps: Optional[np.array] = None,
    sample_size: Optional[Union[int, str]] = None,
    offset: Optional[Union[int, str]] = None,
    correct_scaling_bias: bool = True
) -> Union[float, np.ndarray]:
    """
    Compute the Aït-Sahalia and Jacod (ASJ) Realized Truncated pth Variation for a given series of prices.
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
    if alpha is not None and alpha <= 0:
        raise ValueError("Alpha must be greater than 0.")
    if omega is not None and not (0 <= omega <= 0.5):
        raise ValueError("Omega must be between 0 and 0.5.")
    if p is None or p <= 0:
        raise ValueError("p must be provided and a positive integer.")
    
    N = len(prices) - 1  # Number of returns = number of prices - 1
    if N < 1:
        raise ValueError("At least two prices are required to compute multipower variation.")
    
    biais_scaling = N / (N + 1) if correct_scaling_bias else 1.0
    m_p = mu_x(p)
    
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

        tpvs = np.zeros(len(price_subsamples))
        total_count = 0
        for idx in range(len(price_subsamples)):
            price_sample = price_subsamples[idx]
            timestamp_sample = timestamps_subsamples[idx]
            if len(price_sample) < 2:
                tpvs[idx] = np.nan
            else:
                n = len(price_sample) - 1
                if idx == 0:
                    base_count = n
                total_count += n
                returns = np.diff(np.log(price_sample))
                delta = get_time_delta(timestamps=timestamp_sample, N=n)
                if alpha is None:
                    alpha_truncation = np.where(np.abs(returns) <= delta**(0.5), 1, 0)
                    alpha = 5 * np.sqrt( np.sum(np.abs(returns)**p) * alpha_truncation )
                truncation = np.where(np.abs(returns) <= alpha * (delta**omega), 1, 0)
                tpvs[idx] = ( (delta**(1 - p * 0.5)) / m_p ) * np.sum( np.abs(returns)**p * truncation )

        return biais_scaling * np.sum(tpvs) * (base_count / total_count)

    returns = np.diff(np.log(prices))
    delta = get_time_delta(timestamps=timestamps, N=len(returns))
    if alpha is None:
        alpha_truncation = np.where(np.abs(returns) <= delta**(0.5), 1, 0)
        if beta is None:
            beta = np.sum(np.abs(returns)**p) * alpha_truncation
        alpha = 5 * (beta**0.5)
    truncation = np.where(np.abs(returns) <= alpha * (delta**omega), 1, 0)

    return  biais_scaling * ( (delta**(1 - p * 0.5)) / m_p ) * np.sum( np.abs(returns)**p * truncation )