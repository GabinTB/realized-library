from typing import Optional, Union
import numpy as np
from pandas import to_timedelta
from realized_library.utils.subsampling import compute as subsample

def compute(
    prices: np.ndarray,
    timestamps: Optional[np.ndarray] = None, 
    sample_size: Union[str, int, None] = None,
    offset: Union[str, int, None] = None,
) -> float:
    """
    Compute the realized Minimum Realized Quarticity (MinRQ) from price data.
    "Jump-robust volatility estimation using nearest neighbor truncation"
        by Andersen et al. (2012).
        DOI: 10.1016/j.jeconom.2012.01.011

    Parameters
    ----------
    prices : np.ndarray
        Array of asset prices.
    timestamps : np.ndarray
        Array of timestamps corresponding to the prices.
    resampling_freq : str, optional
        Frequency for resampling the data (e.g., '1min', '5min'). If None, no resampling is performed.
    resampling_size : int, optional
        Size of the resampling window. If None, defaults to 1.

    Returns
    -------
    float
        The computed realized variance.
    """
    n = len(prices)
    if n < 2:
        raise ValueError("At least two prices are required to compute the realized variance.")
    
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

        medRVs = []
        for sample in price_subsamples:
            if len(sample) < 2:
                continue
            medRVs.append(compute(sample, None, None, None))

        return np.mean(medRVs)
    
    else:
        N = len(prices)
        if N < 2:
            raise ValueError("At least two prices are required to compute the MedRV.")
        returns = np.diff(np.log(prices))
        matrix = np.column_stack([returns[:-1], returns[1:]])  # Rolling window: each row has returns^2 at t, t+1
        
        return (np.pi * N / (3 * np.pi - 8) ) * (N / (N - 1)) * np.sum(np.min(np.abs(matrix), axis=1)**4)