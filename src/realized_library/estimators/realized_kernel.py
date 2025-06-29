from functools import singledispatch
from dataclasses import dataclass
import numpy as np
from typing import Optional, Literal, Union
from numba import njit
from bisect import bisect_left
from scipy.optimize import minimize_scalar
from realized_library.estimators.realized_variance import compute as rv
from realized_library.estimators.noise_variance import optimal_q
from realized_library.estimators.integrated_variance import compute as noise_variance
from realized_library.estimators.integrated_variance import compute as integrated_variance

# Ressources:
# - http://dx.doi.org/10.2139/ssrn.620203
# - https://github.com/jonathancornelissen/highfrequency
# - https://web.archive.org/web/20220903214102/https://realized.oxford-man.ox.ac.uk/documentation/econometric-methods

@dataclass
class _KernelPropertises:
    kappa__0: float
    kappa00: float

    def _compute_c_star(self) -> float:
        return ( (self.kappa__0 ** 2) / self.kappa00 ) ** (1/5)

    def __post_init__(self):
        if not all(isinstance(x, float) for x in self.__dict__.values()):
            raise TypeError("All moments must be numeric values.")
        self.c_star: float = self._compute_c_star()

# TODO: check moments for Tukey-Hanning kernels
_MOMENTS = {
    'parzen': _KernelPropertises(12, 0.269),
    'bartlett': _KernelPropertises(0, 1/3),
    'tuckey-hanning': _KernelPropertises(None, 0.375),
    'm-tuckey-hanning-2': _KernelPropertises(None, 0.218),
    'm-tuckey-hanning-5': _KernelPropertises(None, 0.097),
    'm-tuckey-hanning-10': _KernelPropertises(None, 0.05),
    'm-tuckey-hanning-16': _KernelPropertises(None, 0.032),
    'epanechnikov': _KernelPropertises(2, 8/15),
    'second-order': _KernelPropertises(2, 1/5),
    'cubic': _KernelPropertises(6, 0.371),
    'bnhls2008': _KernelPropertises(1, 5/4),
    'quadratic-spectral': _KernelPropertises(1/5, 3*np.pi/5),
}
_KERNELS = _MOMENTS.keys()
    
def _bartlett(x: float) -> float: # Flat-top kernel
    return 1 - abs(x) if abs(x) <= 1 else 0.0

def _epanechnikov(x: float) -> float: # Flat-top kernel
    return 1 - x**2

def _second_order(x: float) -> float: # Flat-top kernel
    return 1 - 2 * x**2 + x**2

def _cubic(x: float) -> float: # Non-Flat-top kernel
    return 1 - 3 * x**2 + 2 * x**3

def _parzen(x: float) -> float: # Non-Flat-top kernel
    if 0 <= x <= 0.5:
        return 1 - 6 * x**2 + 6 * x**3
    elif 0.5 < x <= 1:
        return 2 * (1 - x)**3
    else:
        return 0.0
    
def _tuckey_hanning(x: float) -> float: # Non-Flat-top kernel
    return 0.5 * (1 + np.cos(np.pi * x)) if abs(x) <= 1 else 0.0

def _modified_tuckey_hanning(x: float, power: Literal[2,5,16]) -> float: # Non-Flat-top kernel ; when power = 1, this is the usual Tuckey-Hanning
    return np.sin( np.pi / np.power(2 * (1 - x), power) ) ** 2

def _quadratic_spectral(x: float) -> float: # Non-Flat-top kernel
    return (3/(x**2)) * ( np.sin(x)/(x**2) - np.cos(x) ) if x >= 0 else 0.0

def _bnhls2008(x: float) -> float: # Non-Flat-top kernel
    return (1 + x) * np.exp(-x) if x >= 0 else 0.0

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

def _optimal_c_star(
    kernel: str,
) -> float:
    return _MOMENTS[kernel].c_star if kernel in _MOMENTS else None

def optimal_bandwidth(
    kernel: str,
    n: int,
    omega2_est: float,
    iv_est: float
) -> float:
    """
    Compute the optimal bandwidth for the specified kernel type.
    This function estimates the optimal bandwidth based on the noise variance and integrated variance.

    Parameters
    ----------
    kernel : str
        Type of kernel to use for the estimation. Supported kernels available with `list_kernels()`.
    n : int
        Number of observations (length of the price series).
    omega2_est : float
        Estimated noise variance (ω^2).
    iv_est : float
        Estimated integrated variance (IV).

    Returns
    -------
    float
        Optimal bandwidth for the specified kernel type.
    
    Raises
    ------
    ValueError
        If the kernel type is not supported or if the number of observations is less than 1.
    """
    if kernel not in _KERNELS:
        raise ValueError(f"Invalid kernel type. Supported kernels are: {', '.join(_KERNELS)}")
    if n < 1:
        raise ValueError("Number of observations (n) must be at least 1.")
    c_star = _optimal_c_star(kernel=kernel)
    zeta2_est = omega2_est / iv_est
    zeta = np.sqrt(zeta2_est)
    return c_star * zeta**(4/5) * n**(3/5)

def compute(
    prices: np.ndarray,
    kernel: str = 'parzen',
    bandwidth: Union[int, None] = None,
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
        Type of kernel to use for the estimation. Supported kernels available with `list_kernels()`.
    dof_adjustment : bool, optional
        If True, applies degrees of freedom adjustment to the realized variance estimate.

    Returns
    -------
    float
        Realized variance of the price series.

    Raises
    ------
    ValueError
        If the prices array contains non-positive values.
    ValueError
        If the bandwidth exceeds the number of returns.
    """
    if np.any(prices <= 0):
        raise ValueError("Prices must be strictly positive for log-return calculation.")

    returns: np.array = np.diff(np.log(prices))
    n = len(returns)

    if bandwidth is None:
        bandwidth = n**(3/5) # Rule of thumb for all bandwidth
    elif isinstance(bandwidth, int) and bandwidth < 1:
        raise ValueError("Bandwidth must be a positive integer or 'opt'.")

    if kernel not in _KERNELS:
        raise ValueError(f"Invalid kernel type. Supported kernels are: {', '.join(_KERNELS)}")
    elif kernel == 'parzen':
        k: callable = _parzen
    elif kernel == 'bartlett':
        k: callable = _bartlett
    elif kernel == 'tuckey-hanning':
        k: callable = _tuckey_hanning
    elif kernel.startswith('m-tuckey-hanning'):
        power = int(kernel.split('-')[-1])
        k: callable = lambda x: _modified_tuckey_hanning(x, power)
    elif kernel == 'epanechnikov':
        k: callable = _epanechnikov
    elif kernel == 'cubic':
        k: callable = _cubic
    elif kernel == 'second-order':
        k: callable = _second_order
    elif kernel == 'bnhls2008':
        k: callable = _bnhls2008
    elif kernel == 'quadratic-spectral':
        k: callable = _quadratic_spectral

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

    gamma_pos = np.array([np.dot(returns[:n - h], returns[h:]) for h in range(H + 1)])
    gamma_neg = np.array([np.dot(returns[h:], returns[:n - h]) for h in range(H + 1)])

    weights = np.empty(H + 1)
    weights[0] = 1.0
    weights[1:] = np.array([k((h - 1) / H) for h in range(1, H + 1)])

    if dof_adjustment:
        adj_factors = n / (n - np.arange(H + 1))
    else:
        adj_factors = np.ones(H + 1)

    rk = np.sum(weights * adj_factors * (
        gamma_pos + np.where(np.arange(H + 1) == 0, 0.0, gamma_neg)
    ))

    return rk