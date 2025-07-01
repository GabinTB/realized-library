import numpy as np
import pandas as pd
from realized_library.utils.subsampling import compute as subsample
from realized_library.estimators.realized_variance import compute as rv

def compute(
    prices: np.ndarray,
    timestamps: np.ndarray,
    sample_size: str = "20m",
    offset: str = "1s",
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

    sample_size_ns = int(pd.to_timedelta(sample_size).total_seconds() * 1e9)
    offset_ns = int(pd.to_timedelta(offset).total_seconds() * 1e9)
    if sample_size_ns % offset_ns != 0:
        raise ValueError(f"Sample size {sample_size} must be a multiple of offset {offset} to reduce computation time.")
    nb_samples = sample_size_ns // offset_ns

    prices_subsamples, _ = subsample(
        prices=prices, 
        timestamps=timestamps, 
        sample_size=sample_size, 
        offset=offset,
        nb_samples=nb_samples
    )
    
    rv_parses = []
    for sample in prices_subsamples:
        if len(sample) < 2:
            continue
        rv_parse_i = rv(sample)
        rv_parses.append(rv_parse_i)

    return np.mean(rv_parses) # = IV estimate