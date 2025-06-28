import numpy as np
from typing import Optional, Literal, Union
from numba import njit
from bisect import bisect_left
from realized_library.estimators.realized_variance import compute as rv
from realized_library.estimators.noise_variance import optimal_q
from realized_library.estimators.integrated_variance import compute as noise_variance
from realized_library.estimators.integrated_variance import compute as integrated_variance

# Ressources:
# - http://dx.doi.org/10.2139/ssrn.620203
# - https://github.com/jonathancornelissen/highfrequency
# - https://web.archive.org/web/20220903214102/https://realized.oxford-man.ox.ac.uk/documentation/econometric-methods

_KERNELS = [
    'parzen', 
    'bartlett', 
    'tuckey-hanning', 
    'm-tuckey-hanning-2', 
    'm-tuckey-hanning-5', 
    'm-tuckey-hanning-16', 
    'epanechnikov', 
    # 'second-order', 
    # 'cubic',
    # 'optimal'
]
    
def _bartlett(x: float) -> float: # Flat-top kernel
    return 1 - abs(x) if abs(x) <= 1 else 0.0

def _epanechnikov(x: float) -> float: # Flat-top kernel
    return 1 - x**2

def _second_order(x: float) -> float: # Flat-top kernel
    return 1 - 2 * x**2 + x**2

def _cubic(x: float) -> float: # Flat-top kernel
    return 1 - 3 * x**2 + 2 * x**3

def _parzen(x: float) -> float: # Flat-top kernel
    if 0 <= x <= 0.5:
        return 1 - 6 * x**2 + 6 * x**3
    elif 0.5 < x <= 1:
        return 2 * (1 - x)**3
    else:
        return 0.0
    
def _tuckey_hanning(x: float, power: Literal[1,2,5,16]) -> float: # Flat-top kernel
    return 0.5 * (1 + np.cos(np.pi / (1 - x)**power))

def _modified_tuckey_hanning(x: float, power: Literal[2,5,16]) -> float: # Flat-top kernel ; when power = 1, this is the usual Tuckey-Hanning
    return np.sin( np.pi / np.power(2 * (1 - x), power) ) ** 2

def _optimal(x: float) -> float:
    return (1 - x) * np.exp(-x)

def autocovariance(x: np.ndarray, h: int) -> float:
    if h >= len(x):
        return 0.0
    return np.dot(x[h:], x[:-h])

def list_kernels() -> list[str]:
    """
    List all available kernel types for realized variance estimation.

    Returns
    -------
    list[str]
        List of supported kernel types.
    """
    return _KERNELS

def parzen_optimal_bandwidth(
    prices: np.ndarray,
    timestamps: np.ndarray,
    q: Optional[int] = None,
) -> float:
    """
    Compute the optimal bandwidth for the Parzen kernel using the closed-form solution.
    This function estimates the optimal bandwidth based on the noise variance and integrated variance.

    Parameters
    ----------
    prices : np.ndarray
        Array of strictly positive price observations.
    timestamps : np.ndarray
        Corresponding timestamps in nanoseconds (must be sorted, length must match prices).
    q : Optional[int], optional
        The q parameter for robust ω^2 estimation. If None, it will be estimated from timestamps.

    Returns
    -------
    float
        Optimal bandwidth for the Parzen kernel.

    Raises
    ------
    ValueError
        If prices are not strictly positive or if timestamps are not in nanoseconds.
    """
    if np.any(prices <= 0):
        raise ValueError("Prices must be strictly positive for log-return calculation.")
    
    if q is None:
        q = optimal_q(timestamps)

    omega2_est = noise_variance(prices, q)
    iv_est = integrated_variance(prices, timestamps)
    
    zeta2_est = omega2_est / iv_est  # zeta^2 = ω^2 / IV
    zeta = np.sqrt(zeta2_est)
    
    c_star: float = ((12)**2 / 0.269 )**(1/5) # = 3.5134
    n = len(prices)
    
    return c_star * zeta**(4/5) * n**(3/5)  # Optimal bandwidth for Parzen kernel
    

def compute(
    prices: np.ndarray,
    kernel: str = 'parzen',
    bandwidth: Union[int, Literal['opt', None]] = None,
    dof_adjustment: bool = True
) -> float:
    """
    Compute the realized variance (sum of squared log returns) from high-frequency prices.

    Parameters
    ----------
    prices : np.ndarray
        Array of strictly positive price observations.
    bandwidth : Union[int, Literal['opt', None]], optional
        Bandwidth for the kernel estimator. If 'opt', the optimal bandwidth specific to the chosen kernel will be computed.
        If None, a default bandwidth will be used based on the number of returns.
    kernel : str, optional
        Type of kernel to use for the estimation. Supported kernels are:
        - 'parzen': Parzen kernel (default)
        - 'bartlett': Bartlett kernel
        - 'tuckey-hanning': Tuckey-Hanning kernel
        - 'm-tuckey-hanning-2': Modified Tuckey-Hanning kernel
        - 'm-tuckey-hanning-5': Modified Tuckey-Hanning kernel
        - 'm-tuckey-hanning-16': Modified Tuckey-Hanning kernel
        - 'epanechnikov': Epanechnikov kernel
    dof_adjustment : bool, optional
        If True, applies degrees of freedom adjustment to the realized variance estimate.

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

    returns: np.array = np.diff(np.log(prices))
    n = len(returns)

    if bandwidth is None:
        bandwidth = n**(3/5) # Rule of thumb for all bandwidth

    if kernel not in _KERNELS:
        raise ValueError(f"Invalid kernel type. Supported kernels are: {', '.join(_KERNELS)}")
    elif kernel == 'parzen':
        k: callable = _parzen
        is_flat_top = True
        if bandwidth == 'opt':
            bandwidth = parzen_optimal_bandwidth(prices, timestamps=None)
    elif kernel == 'bartlett':
        k: callable = _bartlett
        is_flat_top = True
        if bandwidth == 'opt':
            # TODO: Implement optimal bandwidth for Bartlett kernel
            raise NotImplementedError("Optimal bandwidth for Bartlett kernel is not implemented yet.")
    elif kernel == 'tuckey-hanning':
        k: callable = _tuckey_hanning
        is_flat_top = True
        if bandwidth == 'opt':
            # TODO: Implement optimal bandwidth for Tuckey-Hanning kernel
            raise NotImplementedError("Optimal bandwidth for Tuckey-Hanning kernel is not implemented yet.")
    elif kernel.startswith('m-tuckey-hanning'):
        power = int(kernel.split('-')[-1])
        k: callable = lambda x: _modified_tuckey_hanning(x, power)
        is_flat_top = True
        if bandwidth == 'opt':
            # TODO: Implement optimal bandwidth for Modified Tuckey-Hanning kernel
            raise NotImplementedError(f"Optimal bandwidth for Modified Tuckey-Hanning kernel with power {power} is not implemented yet.")
    elif kernel == 'epanechnikov':
        k: callable = _epanechnikov
        is_flat_top = True
        if bandwidth == 'opt':
            # TODO: Implement optimal bandwidth for Epanechnikov kernel
            raise NotImplementedError("Optimal bandwidth for Epanechnikov kernel is not implemented yet.")

    H = int(bandwidth)
    if H > n:
        raise ValueError(f"Bandwidth {H} exceeds the number of returns {n}. Please choose a smaller bandwidth.")

    # hs = np.arange(-H, H + 1)
    # # weights = np.array([k(h / (H + 1)) for h in hs]) if is_flat_top else weights = # TODO
    # weights = np.array([k((h - 1) / H) for h in hs]) if is_flat_top else weights = # TODO
    # gamma_h = np.array([
    #     np.dot(returns[abs(h):], returns[:-abs(h)]) if h != 0 else np.dot(returns, returns)
    #     for h in hs
    # ])
    # rk = np.sum(weights * gamma_h)

    rk = autocovariance(returns, 0) # gamma_0
    for h in range(1, H + 1):
        gamma_h = autocovariance(returns, h)
        weight = k(h / H) if is_flat_top else k(h / (H + 1))
        rk += 2 * weight * gamma_h

    if dof_adjustment and n > 1:
        rk *= n / (n - 1)
    
    return rk