import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple
# from numba import njit
from realized_library.utils.resampling import compute as resample

# @njit
# def _obs_based_subsample(prices: np.ndarray, sample_size: int, shift_obs: int) -> List[np.ndarray]:
#     max_start = len(prices) - sample_size
#     results = []
#     for start in range(0, max_start + 1, shift_obs):
#         results.append(prices[start : start + sample_size].copy())
#     return results

# @njit
# def _searchsorted(a: np.ndarray, v: int, side: str = 'left') -> int:
#     low = 0
#     high = len(a)
#     while low < high:
#         mid = (low + high) // 2
#         if (side == 'left' and a[mid] < v) or (side == 'right' and a[mid] <= v):
#             low = mid + 1
#         else:
#             high = mid
#     return low

# @njit
# def _time_based_subsample(prices: np.ndarray, timestamps: np.ndarray, sample_duration: int, shift_time: int) -> List[np.ndarray]:
#     n = len(prices)
#     start_idx = 0
#     results = []
#     while start_idx < n:
#     # while start_idx < sample_duration // shift_time:
#         window_start_time = timestamps[start_idx]
#         window_end_time = window_start_time + sample_duration

#         # Binary search for end_idx
#         end_idx = _searchsorted(timestamps, window_end_time, side='left')

#         if end_idx > start_idx:
#             results.append(prices[start_idx:end_idx].copy())

#         # Binary search for next start_idx (shifted window start)
#         next_window_start_time = window_start_time + shift_time
#         start_idx = _searchsorted(timestamps, next_window_start_time, side='left')

#     return results

# def compute(
#     prices: np.ndarray,
#     timestamps: Optional[np.ndarray] = None,
#     sample_size: Optional[int] = None,
#     shift_obs: Optional[int] = None,
#     sample_duration: Optional[int] = None,
#     shift_time: Optional[int] = None
# ) -> List[np.array]:
#     """
#     High-performance overlapping subsampling (obs-based or time-based), nanosecond-precision, Numba accelerated.
#     """
#     if not isinstance(prices, np.ndarray) or prices.ndim != 1:
#         raise ValueError("'prices' must be a 1D numpy array.")

#     is_obs_based = sample_size is not None and shift_obs is not None
#     is_time_based = sample_duration is not None and shift_time is not None and timestamps is not None

#     if is_obs_based and is_time_based:
#         raise ValueError("Specify either observation-based OR time-based subsampling, not both.")

#     if not is_obs_based and not is_time_based:
#         raise ValueError("You must specify either (sample_size and shift_obs) or (sample_duration, shift_time, timestamps).")

#     if is_obs_based:
#         if shift_obs <= 0 or sample_size <= 0:
#             raise ValueError("Both 'sample_size' and 'shift_obs' must be positive integers.")
#         if sample_size % shift_obs != 0:
#             raise ValueError("'shift_obs' must divide 'sample_size' exactly to reduce number of samples.")
#         return _obs_based_subsample(prices, sample_size, shift_obs)

#     else:
#         if shift_time <= 0 or sample_duration <= 0:
#             raise ValueError("Both 'sample_duration' and 'shift_time' must be positive integers.")
#         if sample_duration % shift_time != 0:
#             raise ValueError("'shift_time' must divide 'sample_duration' exactly to reduce number of samples.")
#         if timestamps.shape != prices.shape:
#             raise ValueError("'timestamps' and 'prices' must have the same shape.")
#         return _time_based_subsample(prices, timestamps, sample_duration, shift_time)

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
        # raise ValueError("Both 'sample_size' and 'offset' must be of the same type (either both int or both str).")
        warnings.warn("Both 'sample_size' and 'offset' should be of the same type, otherwise nothing ensures that offset will be lower than sample_size.")
    if isinstance(sample_size, str) and isinstance(offset, str):
        if pd.to_timedelta(sample_size) < pd.to_timedelta(offset):
            raise ValueError("'sample_size' must be greater than or equal to 'offset' when both are strings.")
    elif isinstance(sample_size, int) and isinstance(offset, int):
        if sample_size < offset:
            raise ValueError("'sample_size' must be greater than or equal to 'offset' when both are integers.")

    _, time_grid = resample(prices, timestamps, offset, explicit_start=None, explict_end=None, ffill=False)
    
    prices_subsamples = []
    timestamps_subsamples = []
    for start in time_grid:
        temp_prices, temp_timestamps = resample(prices, timestamps, sample_size, explicit_start=start, explict_end=None, ffill=True)
        prices_subsamples.append(temp_prices)
        timestamps_subsamples.append(temp_timestamps)
        if nb_samples is not None and len(prices_subsamples) >= nb_samples:
            break

    return prices_subsamples, timestamps_subsamples