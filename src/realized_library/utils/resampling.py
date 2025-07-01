import numpy as np
import re
import polars as pl
from typing import Tuple, Union


def _is_valid_high_freq_interval(interval: str) -> bool:
    """
    Validate if the input interval string is a valid high-frequency interval
    between 1 nanosecond and 60 minutes (1 hour), suitable for Polars resampling.

    Allowed units: ns, us, ms, s, m

    Parameters
    ----------
    interval : str
        The interval string to validate (e.g., '1s', '500ms', '10m').

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    pattern = r'^\s*(\d+)\s*(ns|us|ms|s|m)\s*$'
    match = re.match(pattern, interval)
    if not match:
        return False

    value = int(match.group(1))
    unit = match.group(2)

    # Apply limits per unit
    if unit == 'ns' and value >= 1:
        return True
    elif unit == 'us' and value >= 1:
        return True
    elif unit == 'ms' and value >= 1:
        return True
    elif unit == 's' and value >= 1:
        return True
    elif unit == 'm' and 1 <= value <= 60:
        return True
    else:
        return False

def compute(
    prices: np.ndarray,
    timestamps: np.ndarray,
    sample_size: Union[int, str],
    explicit_start: int = None,
    explict_end: int = None,
    ffill: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample high-frequency price series using the last price in each time bin, with optional forward fill.

    Parameters
    ----------
    timestamps : np.ndarray
        Nanosecond timestamps (same length as prices).
    prices : np.ndarray
        Prices (same length as timestamps).
    sample_size : Union[int, str]
        Resample frequency (e.g., '1s', '5m', etc.) if str or number of observations per bin if int.
    explicit_start : int, optional
        Explicit start timestamp in nanoseconds.
    explict_end : int, optional
        Explicit end timestamp in nanoseconds.
    ffill : bool, optional
        Whether to forward-fill missing bins. Default is False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Resampled prices and timestamps (both np.ndarray).
    """
    if timestamps.shape != prices.shape:
        raise ValueError("'timestamps' and 'prices' must have the same shape.")
    if timestamps.ndim != 1:
        raise ValueError("'timestamps' and 'prices' must be 1D arrays.")
    if isinstance(sample_size, str) and not _is_valid_high_freq_interval(sample_size):
        raise ValueError(f"Invalid sample_size: '{sample_size}'. Must be a valid interval from 1ns to 60m (e.g., '10us', '1s', '500ms', '10m').")

    start_ns = explicit_start if explicit_start is not None else timestamps[0]
    end_ns = explict_end if explict_end is not None else timestamps[-1]

    mask = (timestamps >= start_ns) & (timestamps <= end_ns)
    timestamps = timestamps[mask]
    prices = prices[mask]

    if isinstance(sample_size, str):
        df = pl.DataFrame({
            "timestamp": pl.from_numpy(timestamps).cast(pl.Datetime("ns")),
            "price": pl.from_numpy(prices)
        })
        resampled_df = (
            df.sort("timestamp")
            .group_by_dynamic(
                index_column="timestamp",
                every=sample_size,
                #   period=resample_freq, # Useless since it is automatically assigned to `every`
                closed="left",
                start_by="datapoint",
            )
            .agg(pl.col("price").last().alias("last_price"))
            .sort("timestamp")
        )
        if ffill:
            resampled_df = resampled_df.with_columns(pl.col("last_price").fill_null(strategy="forward"))
        # resampled_df = resampled_df.drop_nulls(subset=["last_price"]) # Drop any bins still null (only possible for leading bins if ffill=True)
        resampled_timestamps = resampled_df["timestamp"].cast(pl.Int64).to_numpy()
        resampled_prices = resampled_df["last_price"].to_numpy()

    elif isinstance(sample_size, int):
        if sample_size < 1:
            raise ValueError("sample_size must be a positive integer.")
        resampled_prices = prices[::sample_size]
        resampled_timestamps = timestamps[::sample_size]
    
    else:
        raise ValueError("sample_size must be either a string (e.g., '1s', '5m') or an integer.")
    
    if len(resampled_prices)!= len(resampled_timestamps):
        raise ValueError("Error in resampling: prices and timestamps lengths do not match after resampling.")

    return resampled_prices, resampled_timestamps
