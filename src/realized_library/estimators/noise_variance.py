import numpy as np
from typing import Optional, Literal
from realized_library.estimators.realized_variance import compute as rv
from realized_library.estimators.realized_kernel import compute as rk

def _rv_sparse(
    prices: np.ndarray,
    debiasing: Literal['bandi-russel', 'oomen', 'bnhls', None] = 'bandi-russel',
    bnhls_debiasing_params: Optional[dict] = None
) -> float:
    """
    Estimate the realized variance from a sparse set of price observations.
    This function computes the realized variance from a subsample of prices, applying debiasing methods
    if specified. The debiasing methods include Bandi-Russel, Oomen, and Bandi-Nielsen-Hansen-Lunde-Shephard (BNHLS).

    Parameters
    ----------
    prices : np.ndarray
        Array of strictly positive price observations.
    debiasing : Literal['bandi-russel', 'oomen', 'bnhls', None], optional
        Debiasing method to use for the realized variance estimation. Default is 'bandi-russel'.
        Options:
        - 'bandi-russel': Bandi-Russel debiasing method.
        - 'oomen': Oomen debiasing method.
        - 'bnhls': BNHLS debiasing method (not yet implemented).
        - None: No debiasing applied.
    bnhls_debiasing_params : dict, optional
        Parameters for the Bandi-Nielsen-Hansen-Lunde-Shephard (BNHLS) debiasing method.
        Must include 'resampling_freq', 'bandwidth' (optional), and 'kernel'.
    
    Returns
    -------
    float
        Estimated realized variance.
    
    Raises
    ------
    ValueError
        If the prices array has less than two observations.
        If an unsupported debiasing method is specified.
        If BNHLS debiasing parameters are not provided or incomplete.
    """
    if len(prices) < 2:
        raise ValueError("At least two price observations are required to compute noise variance.")

    variance = rv(prices)
    if debiasing is None:
        return variance
    elif debiasing == 'bandi-russel':
        log_returns = np.diff(np.log(prices))
        n = np.sum(log_returns != 0)
        if n == 0:
            return 0.0
        return variance / (2 * n)
    elif debiasing == 'oomen':
        log_returns = np.diff(np.log(prices))
        n = len(log_returns)
        return ( -1/(n-1) ) * np.dot(log_returns[:-1], log_returns[1:])
    elif debiasing == 'bnhls':
        if bnhls_debiasing_params is None:
            raise ValueError("Bandi-Nielsen-Hansen-Lunde-Shephard (BNHLS) debiasing requires parameters to be provided.")
        resampling_freq = bnhls_debiasing_params.get('resampling_freq', None)
        bandwidth = bnhls_debiasing_params.get('bandwidth', len(resampling_freq) ** (1/3))
        kernel = bnhls_debiasing_params.get('kernel', None)
        if resampling_freq is None or bandwidth is None or kernel is None:
            raise ValueError("Bandi-Nielsen-Hansen-Lunde-Shephard (BNHLS) debiasing requires 'resampling_freq', 'bandwidth', and 'kernel' parameters.")
        noise_variance = _rv_sparse(prices=prices, debiasing='bandi-russel')
        resampled_prices = prices[::resampling_freq]
        med_freq_rk = rk(resampled_prices, bandwidth=bandwidth, kernel=kernel)
        med_freq_rv = rv(resampled_prices)
        return np.exp( np.log(noise_variance) - med_freq_rk/med_freq_rv )
    else:
        raise ValueError(f"Unsupported debiasing method: {debiasing}. Supported methods are 'bandi-russel', 'oomen', 'bnhls', or None.")

def optimal_q(
    timestamps: np.ndarray,
    target_time_interval: int = 120,  # Minimum time interval in seconds (default is 2 minutes)
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

def compute(
    prices: np.ndarray,
    q: Optional[float] = None,
    debiasing: Literal['bandi-russel', 'oomen', 'bnhls', None] = 'bandi-russel',
    bnhls_debiasing_params: dict = { 
        'resampling_size': 100,
        'bandwidth': 10,
        'kernel': 'bartlett'
    }
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
    prices : np.ndarray
        Array of strictly positive price observations.
    q : Optional[float], optional
        The q parameter for robust ω^2 estimation. If None, it will be estimated from
    debiasing : Literal['bandi-russel', 'oomen', 'bnhls', None], optional
        Debiasing method to use for the realized variance estimation. Default is 'bandi-russel'.
        Options:
        - 'bandi-russel': Bandi-Russel debiasing method.
        - 'oomen': Oomen debiasing method.
        - 'bnhls': BNHLS debiasing method (not yet implemented).
        - None: No debiasing applied.

    Returns
    -------
    float
        Estimated omega2 parameter.

    Raises
    ------
    ValueError
        If q is less than 2, as robust ω^2 estimation requires q > 1.
        If q is None, as it must be provided for robust ω^2 estimation.
        If the number of distinct realized variances is not equal to q - 1.
    """
    if q is not None and q < 2:
        raise ValueError("q must be greater to 1 for robust ω^2 estimation.")
    if q is None:
        raise ValueError("q must be provided for robust ω^2 estimation. Use optimal_q() to estimate it.")

    noise_estimates = []
    for i in range(2, q+1):
        subsample_prices = prices[::i] # Subsample every q-th observation starting at offset i
        noise_estimate_i = _rv_sparse( # RV_sparse_i
            prices=subsample_prices,
            debiasing=debiasing,
            bnhls_debiasing_params=bnhls_debiasing_params
        )
        noise_estimates.append(noise_estimate_i)
    
    if len(noise_estimates) != q - 1:
        raise ValueError("We didn't obtain q distinct realised variances")

    return np.mean(noise_estimates)