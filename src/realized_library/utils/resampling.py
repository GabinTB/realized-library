import numpy as np
from bisect import bisect_left
from numba import njit
from typing import Tuple


def _convert_timestamps_to_ns(timestamps: np.ndarray) -> np.array:
    """
    Detect the unit of the timestamps based on the first two values and convert them to nanoseconds.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamps (numeric).

    Returns
    -------
    np.array
        Timestamps converted to nanoseconds.

    Raises
    ------
    ValueError
        If timestamps are not numeric or have inconsistent units.
    """
    # timestamp_lenghts = [len(str(int(ts))) for ts in timestamps]
    # if np.unique(timestamp_lenghts).size != 1:
    #     raise ValueError("Timestamps must have a consistent unit.")
    # lenght = timestamp_lenghts[0]
    
    lenght = len(str(int(timestamps[0])))
    if lenght == 10: # unit = seconds
        return timestamps * 1e9
    elif lenght == 13: # unit = milliseconds
        return timestamps * 1e6
    elif lenght == 16: # unit = microseconds
        return timestamps * 1e3
    elif lenght == 19: # unit = nanoseconds
        return timestamps
    else:
        raise ValueError(f"Invalid timestamp length: {lenght}. Timestamps must be in seconds, milliseconds, microseconds, or nanoseconds.")


def _parse_resample_freq(freq_str: str) -> float:
    """
    Convert a resampling frequency string to seconds.
    """
    if not isinstance(freq_str, str):
        raise ValueError(f"Invalid resample frequency: input must be a string, got {type(freq_str)}.")

    freq_str = freq_str.strip().lower()
    units = {
        'ns': 1e-9,
        'us': 1e-6,
        'ms': 1e-3,
        's': 1,
        'm': 60
    }

    for unit, factor in units.items():
        if freq_str.endswith(unit):
            try:
                value = float(freq_str.replace(unit, ''))
                return value * factor
            except ValueError:
                break
    raise ValueError(f"Invalid resample frequency string: {freq_str}")


@njit
def _get_last_prices_in_bins(
    timestamps: np.ndarray,
    prices: np.ndarray,
    bin_edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = len(bin_edges) - 1
    bin_timestamps = []
    bin_prices = []
    n_obs = len(timestamps)

    obs_idx = 0

    for bin_idx in range(n_bins):
        bin_start = bin_edges[bin_idx]
        bin_end = bin_edges[bin_idx + 1]

        # Binary search for start index (skip points before bin_start)
        left = obs_idx
        right = n_obs
        while left < right:
            mid = (left + right) // 2
            if timestamps[mid] < bin_start:
                left = mid + 1
            else:
                right = mid
        start_idx = left

        # Binary search for end index (first point >= bin_end)
        left = start_idx
        right = n_obs
        while left < right:
            mid = (left + right) // 2
            if timestamps[mid] < bin_end:
                left = mid + 1
            else:
                right = mid
        end_idx = left

        if end_idx > start_idx:
            last_idx = end_idx - 1
            bin_timestamps.append(timestamps[last_idx])
            bin_prices.append(prices[last_idx])

        obs_idx = end_idx  # Skip to next bin start

    return np.array(bin_timestamps), np.array(bin_prices)


def compute(
    timestamps: np.ndarray,
    prices: np.ndarray,
    resample_freq: str
) -> np.ndarray:
    """
    Resample high-frequency price series using the last price in each time bin.

    Parameters
    ----------
    timestamps : np.ndarray
        Timestamps (numeric, same length as prices).

    prices : np.ndarray
        Prices (strictly positive, same length as timestamps).

    resample_freq : str
        Resampling frequency (e.g., '1s', '5m', etc.).

    Returns
    -------
    np.ndarray
        Resampled price series.
    """
    if not isinstance(timestamps, np.ndarray) or not isinstance(prices, np.ndarray):
        raise ValueError("Both timestamps and prices must be numpy arrays.")
    if len(timestamps) != len(prices):
        raise ValueError("Length of timestamps and prices must match.")
    
    # Sort inputs by timestamps
    sorted_idx = np.argsort(timestamps)
    timestamps_sorted = timestamps[sorted_idx]
    prices_sorted = prices[sorted_idx]

    # Convert resampling frequency to bin size
    timestamps_ns = _convert_timestamps_to_ns(timestamps_sorted)
    interval_nanoseconds = _parse_resample_freq(resample_freq) * 1e9  # Convert to nanoseconds

    # Build bin edges
    start_time = timestamps_ns[0]
    end_time = timestamps_ns[-1]+ interval_nanoseconds  # Ensure the last observation falls inside the last bin
    intervals = np.arange(start_time, end_time, interval_nanoseconds)

    # Call numba-accelerated binning
    return _get_last_prices_in_bins(timestamps_ns, prices_sorted, intervals)
