import numpy as np
from bisect import bisect_left
from realized_library.estimators.realized_variance import compute as rv

def compute(
    prices: np.ndarray,
    timestamps: np.ndarray,
    subsample_time_min: int = 20,
    shift_seconds: int = 1,
) -> float:
    """
    Estimate the integrated variance (IV) using the RVsparse method with multiple shifted grids.

    For each grid shift, the function samples prices at regular grid intervals (default: 20 minutes),
    starting at different 1-second shifts within the first grid interval, computes the realized variance,
    and returns the average across all shifts.

    Parameters
    ----------
    prices : np.ndarray
        Array of strictly positive price observations.
    timestamps : np.ndarray
        Corresponding timestamps in nanoseconds (must be sorted, length must match prices).
    grid_time_min : int, optional
        Grid interval in minutes (default is 20 minutes).
    shift_seconds : int, optional
        Temporal shift between grids in seconds (default is 1 second).

    Returns
    -------
    float
        Estimated integrated variance (IV).
    
    Raises
    ------
    ValueError
        If timestamps are not in nanoseconds, or if grid_time_min or shift_seconds are out
        of the expected range.
    ValueError
        If the number of shifts is less than 1, which indicates that the timestamp range is
        too small for the given grid_time_min and shift_seconds.
    ValueError
        If no valid grid produced any realized variance.
    """
    if len(str(int(timestamps[0]))) != 19:
        raise ValueError("Timestamps must be in nanoseconds (19 digits).")
    if not 10 <= subsample_time_min <= 60:
        raise ValueError("grid_time should be between 10 and 60 minutes.")
    if not 1 <= shift_seconds <= 10:
        raise ValueError("grid_time should be between 10 and 60 minutes.")

    subsample_time_ns = subsample_time_min * 60 * 1e9  # Convert grid time to nanoseconds
    increment_ns = shift_seconds * 1e9               # Shift increment in nanoseconds
    num_increments = int((subsample_time_min * 60) // shift_seconds)  # Total number of grid shifts (e.g., 1200 for 20min grid with 1s shift)
    if num_increments < 1:
        raise ValueError("The number of shifts must be at least 1. Timestamp range too small for the given subsample_time_min and shift_seconds.")

    total_time_ns = timestamps[-1] - timestamps[0]
    num_grid_points_per_shift = int(total_time_ns // subsample_time_ns) + 1  # Number of grid points (samples) per shift

    rv_parses = []
    for shift_idx in range(num_increments):
        grid_start_time = int(timestamps[0] + shift_idx * increment_ns)

        grid_prices = []
        for point_idx in range(num_grid_points_per_shift):
            target_time = grid_start_time + point_idx * subsample_time_ns
            price_idx = bisect_left(timestamps, target_time)
            if price_idx >= len(prices):
                break
            grid_prices.append(prices[price_idx])

        grid_prices = np.array(grid_prices)
        if len(grid_prices) >= 2:
            # log_returns = np.diff(np.log(grid_prices))
            # rv_parse_i = np.sum(log_returns ** 2)
            rv_parse_i = rv(grid_prices)
            rv_parses.append(rv_parse_i)

    if len(rv_parses) != num_increments:
        raise ValueError(
            f"Expected {num_increments} shifts, but only {len(rv_parses)} valid RVs were computed. "
            "This may indicate that the price series is too short or the grid parameters are too strict."
        )

    return np.mean(rv_parses) # = IV estimate