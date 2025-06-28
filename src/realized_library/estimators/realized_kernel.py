import numpy as np
from typing import Optional, Literal
from numba import njit
from bisect import bisect_left
from realized_library.estimators.realized_variance import compute as rv

# Ressources:
# - http://dx.doi.org/10.2139/ssrn.620203

_KERNELS = [
    'parzen', 
    'bartlett', 
    'tuckey-hanning-1', 
    'tuckey-hanning-2', 
    'tuckey-hanning-5', 
    'tuckey-hanning-16', 
    'epanechnikov', 
    'second-order', 
    'cubic',
    'optimal'
]
    
def _bartlett(x: float) -> float: # Flat-top kernel
    return 1 - abs(x) if abs(x) <= 1 else 0.0

def _epanechnikov(x: float) -> float: # Flat-top kernel
    return 1 - x**2

def _second_order(x: float) -> float: # Flat-top kernel
    return 1 - 2 * x**2 + x**2

def _cubic(x: float) -> float: # Flat-top kernel
    return 1 - 3 * x**2 + 2 * x**3

def _parzen(x: float) -> float: # Non-flat-top kernel
    if 0 <= x <= 0.5:
        return 1 - 6 * x**2 + 6 * x**3
    elif 0.5 < x <= 1:
        return 2 * (1 - x)**3
    else:
        return 0.0

def _tuckey_hanning(x: float, power: Literal[1,2,5,16]) -> float: # Non-flat-top kernel
    # In the case power = 1, this is the usual Tuckey-Hanning kernel: # 0.5 * (1 + np.cos(np.pi * x))
    return np.sin( np.pi / np.power(2 * (1 - x), power) ) ** 2

def _optimal(x: float) -> float:
    return (1 - x) * np.exp(-x)

def estimate_iv(
    prices: np.ndarray,
    timestamps: np.ndarray,
    grid_time_min: int = 20,
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
    if not 10 <= grid_time_min <= 60:
        raise ValueError("grid_time should be between 10 and 60 minutes.")
    if not 1 <= shift_seconds <= 10:
        raise ValueError("grid_time should be between 10 and 60 minutes.")

    grid_interval_ns = grid_time_min * 60 * 1e9  # Convert grid time to nanoseconds
    shift_ns = shift_seconds * 1e9               # Shift increment in nanoseconds
    num_shifts = int((grid_time_min * 60) // shift_seconds)  # Total number of grid shifts (e.g., 1200 for 20min grid with 1s shift)
    if num_shifts < 1:
        raise ValueError("The number of shifts must be at least 1. Timestamp range too small for the given grid_time_min and shift_seconds.")

    total_time_ns = timestamps[-1] - timestamps[0]
    num_grid_points_per_shift = int(total_time_ns // grid_interval_ns) + 1  # Number of grid points (samples) per shift

    rv_parses = []
    for shift_idx in range(num_shifts):
        grid_start_time = int(timestamps[0] + shift_idx * shift_ns)

        grid_prices = []
        for point_idx in range(num_grid_points_per_shift):
            target_time = grid_start_time + point_idx * grid_interval_ns
            price_idx = bisect_left(timestamps, target_time)
            if price_idx > len(prices):
                break
            grid_prices.append(prices[price_idx])

        grid_prices = np.array(grid_prices)
        if len(grid_prices) >= 2:
            log_returns = np.diff(np.log(grid_prices))
            rv_parse_i = np.sum(log_returns ** 2)
            rv_parses.append(rv_parse_i)

    if len(rv_parses) != num_shifts:
        raise ValueError(
            f"Expected {num_shifts} shifts, but only {len(rv_parses)} valid RVs were computed. "
            "This may indicate that the price series is too short or the grid parameters are too strict."
        )

    return np.mean(rv_parses) # = IV estimate


def optimal_q(
    timestamps: np.ndarray,
    target_time_interval: int = 300,  # Minimum time interval in seconds (default is 5 minutes)
) -> int:
    """
    Estimate the optimal q parameter for robust ω^2 estimation.
    This function finds the largest q such that the average time interval between every q-th observation
    is closest to the target time interval (default is 2 minutes = 120 seconds).

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamps corresponding to the log returns. Must be nanoseconds timestamps.
    target_time_interval : int, optional
        Target average time interval in seconds for the estimation (default is 2 minutes = 120 seconds).

    Returns
    -------
    int
        Optimal q parameter for robust ω^2 estimation.
    """
    if len(str(int(timestamps[0]))) != 19:
        raise ValueError("Timestamps must be in nanoseconds (19 digits).")
    if not 1 <= target_time_interval <= 30 * 60:
        raise ValueError("target_time_interval should be between 1s and 30 minutes.")

    lowest_avg_interval_diff = np.inf
    for q in range(2, 10000): # 10000 is an arbitrary upper limit for q, experiments have shown that q rarely exceeds 5000 for very liquid assets
        
        time_diffs = []
        for i in range(q):
            subsampled_ts = timestamps[::q]
            time_diff_ns = subsampled_ts[1] - subsampled_ts[0]
            time_diff_sec = time_diff_ns / 1e9  # Convert nanoseconds to seconds
            time_diffs.append(time_diff_sec)
        avg_interval = np.mean(time_diffs)

        avg_interval_diff = abs(target_time_interval - avg_interval)  # Difference from 2 minutes
        if avg_interval_diff < lowest_avg_interval_diff:
            lowest_avg_interval_diff = avg_interval_diff
        else:
            if q - 1 <= 1:
                raise ValueError(
                    f"q must be greater than 1 for robust ω^2 estimation, but found q = {q - 1}."
                    "Consider increasing the target time interval relative to the event frequencies (timestamps)."
                )
            return q - 1 # q_opt is the last q that had a lower average time difference than the target

    raise ValueError(
        f"Could not find a suitable q for target time interval {target_time_interval} minutes. "
        "Consider adjusting the target time interval."
    )

def estimate_omega2(
    prices: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    q: Optional[float] = None,
) -> float:
    """
    Estimate the omega2 parameter from log returns.
    Estimation method from  https://doi.org/10.1111/j.1368-423X.2008.00275.x:
    computing the realised variance using every q-th trade or quote, leading to q distinct RVs,
    and then averaging these RVs to obtain omega2, with q > 1 for robustness.
    Ideally, choose q such that every q-th observation is, on average, 2 minutes apart, examples:
    - We recommend q = 50 for trade prices of liquid assets
    - We recommend q = 70 for quote prices of liquid assets
    - q = 120 for 1s close prices

    Parameters
    ----------
    log_returns : np.ndarray
        Array of log returns.

    Returns
    -------
    float
        Estimated omega2 parameter.
    """
    if q is not None and q < 2:
        raise ValueError("q must be greater to 1 for robust ω^2 estimation.")
    elif q is None and timestamps is None:
        raise ValueError("Either q or timestamps must be provided to estimate omega2.")
    
    if q is None:
        q = optimal_q(timestamps)

    omega2_estimates = []
    for i in range(2, q+1):
        subsample_prices = prices[::q] # Subsample every q-th observation starting at offset i
        n = len(subsample_prices)
        
        if n < 2:
            continue  # Skip subsamples with too few points
        rv_dense_i = rv(subsample_prices)  # Compute realized variance for this subsample
        omega2_i = rv_dense_i / (2*n)
        omega2_estimates.append(omega2_i)
    
    if len(omega2_estimates) != q - 1:
        raise ValueError("We didn't obtain q distinct realised variances")

    return np.mean(omega2_estimates)

def compute(
    prices: np.ndarray,
    timestamps: np.ndarray,
    # kernel: str = 'parzen',
    omega2: Optional[float] = None,
    q: Optional[int] = None,
    iv: Optional[float] = None,
    bandwidth: Optional[int] = None,
) -> float:
    """
    Compute the realized variance (sum of squared log returns) from high-frequency prices.

    Parameters
    ----------
    prices : np.ndarray
        Array of strictly positive price observations.
    bandwidth : Optional[int], optional
        Bandwidth for the kernel. Default is None for otpimal bandwidth selection via closed-form solution,
        rule-of-thumb, or plug-in method.

    Returns
    -------
    float
        Realized variance of the price series.

    Raises
    ------
    ValueError
        For invalid inputs.
    """
    if np.any(prices <= 0):
        raise ValueError("Prices must be strictly positive for log-return calculation.")
    kernel = 'parzen'  # Default kernel, can be changed if needed to expand to multiple kernels

    log_returns: np.array = np.diff(np.log(prices))
    n = len(log_returns)
    omega2_est = omega2 if omega2 is not None else estimate_omega2(prices, timestamps, q)
    iv_est = iv if iv is not None else estimate_iv(log_returns, timestamps)
    zeta2_est = omega2_est / iv_est  # zeta^2 = ω^2 / IV
    zeta = np.sqrt(zeta2_est)

    if kernel not in _KERNELS:
        raise ValueError(f"Invalid kernel type. Supported kernels are: {', '.join(_KERNELS)}")
    elif kernel == 'parzen':
        k: callable = _parzen
        c_star: float = ((12)**2 / 0.269 )**(1/5) # = 3.5134
        optimal_bandwidth: float = c_star * zeta**(4/5) * n**(3/5)

    H = optimal_bandwidth if bandwidth is None else bandwidth

    hs = np.arange(-H, H + 1)
    weights = np.array([k(h / (H + 1)) for h in hs])
    gamma_h = np.array([
        np.dot(log_returns[abs(h):], log_returns[:-abs(h)]) if h != 0 else np.dot(log_returns, log_returns)
        for h in hs
    ])
    rk = np.sum(weights * gamma_h)
    
    return rk