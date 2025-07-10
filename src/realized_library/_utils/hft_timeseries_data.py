from typing import Optional
import numpy as np

def get_time_delta(
    timestamps: Optional[np.ndarray] = None,
    N: Optional[int] = None # Should be the number of returns (N = len(prices) - 1

):
    if timestamps is not None:
        if not np.all(np.diff(timestamps, n=2) == 0):
            # dt_ns = np.mean(np.diff(timestamps)) # Average time interval (useful for trade/quote prices); if equally spaced, it is simply the sampling interval
            # return dt_ns / (24 * 60 * 60 * 1e9)  # Convert to fraction of day
            return 1.0 / (len(timestamps) - 1) # Since len(timestamps) = len(prices) = N, we have N-1 returns
        else:
            dt_ns = timestamps[1] - timestamps[0] # Sampling interval in nanoseconds
            return dt_ns / (24 * 60 * 60 * 1e9)  # Convert to fraction of day
    elif N is not None:
        return 1.0 / N
    else:
        raise ValueError("Either timestamps or returns must be provided to compute the time delta.")