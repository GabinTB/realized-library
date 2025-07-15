import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple
from realized_library.utils.resampling import compute as resample


def compute(
    prices: np.ndarray,
    timestamps: np.ndarray,
    sample_size: Union[int, str],
    offset: Union[int, str],
    nb_samples: Optional[int] = None
) -> Tuple[List[np.array], List[np.array]]:
    """
    High-performance overlapping subsampling (obs-based or time-based), nanosecond-precision.

    Parameters
    ----------
    prices : np.ndarray
        Array of asset prices.
    timestamps : np.ndarray
        Array of timestamps corresponding to the prices.
    sample_size : Union[int, str]
        Size of the subsample, either as a number of observations (int) or a time interval (str, e.g., '1s', '5m').
    offset : Union[int, str]
        Offset for the subsampling, either as a number of observations (int) or a time interval (str, e.g., '1s', '5m').
    nb_samples : Optional[int]
        Maximum number of subsamples to return. If None, all possible subsamples are returned.
    """
    if ( isinstance(sample_size, str) and isinstance(offset, int) ) \
    or ( isinstance(sample_size, int) and isinstance(offset, str) ):
        raise ValueError("Both 'sample_size' and 'offset' must be of the same type (either both int or both str).")
        # warnings.warn("Both 'sample_size' and 'offset' should be of the same type, otherwise nothing ensures that offset will be lower than sample_size.")
    if isinstance(sample_size, str) and isinstance(offset, str):
        if pd.to_timedelta(sample_size) <= pd.to_timedelta(offset):
            raise ValueError("'sample_size' must be greater than 'offset' when both are strings.")
        expected_samples_lenght = int(pd.to_timedelta(sample_size).total_seconds() * 1e9 // pd.to_timedelta(offset).total_seconds() * 1e9)
    elif isinstance(sample_size, int) and isinstance(offset, int):
        if sample_size < offset:
            raise ValueError("'sample_size' must be greater than or equal to 'offset' when both are integers.")
        expected_samples_lenght = sample_size // offset

    _, time_grid = resample(prices, timestamps, offset, explicit_start=None, explict_end=None, ffill=False)
    
    prices_subsamples = []
    timestamps_subsamples = []
    for start in time_grid:
        temp_prices, temp_timestamps = resample(prices, timestamps, sample_size, explicit_start=start, explict_end=None, ffill=True)
        # if len(temp_prices) < 2:
        #     break
        prices_subsamples.append(temp_prices)
        timestamps_subsamples.append(temp_timestamps)
        if nb_samples is not None and len(prices_subsamples) >= nb_samples:
            break

    return prices_subsamples, timestamps_subsamples