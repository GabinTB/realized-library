import warnings
from dataclasses import dataclass
import numpy as np
from typing import Literal, Union, Callable
from realized_library.utils.derivative_approximation import first_derivative, second_derivative
from realized_library.utils.integral_approximation import numerical_integral

# Ressources:
# - http://dx.doi.org/10.2139/ssrn.620203
# - https://github.com/jonathancornelissen/highfrequency
# - https://web.archive.org/web/20220903214102/https://realized.oxford-man.ox.ac.uk/documentation/econometric-methods


########################################################################################
#                                   Kernel definitions                                 #
########################################################################################

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
    elif 0.5 <= x <= 1:
        return 2 * (1 - x)**3
    else:
        return 0.0
    
def _tuckey_hanning(x: float) -> float: # Non-Flat-top kernel
    return 0.5 * (1 + np.cos(np.pi * x)) if abs(x) <= 1 else 0.0

def _modified_tuckey_hanning(x: float, power: Literal[2,5,10,16]) -> float: # When power = 1, this is the usual Tuckey-Hanning
    return np.sin( np.pi / np.power(2 * (1 - x), power) ) ** 2

def _modified_tuckey_hanning_2(x: float) -> float: # Non-Flat-top kernel ; when power = 1, this is the usual Tuckey-Hanning
    return _modified_tuckey_hanning(x, 2)

def _modified_tuckey_hanning_5(x: float) -> float: # Non-Flat-top kernel
    return _modified_tuckey_hanning(x, 5)

def _modified_tuckey_hanning_10(x: float) -> float: # Non-Flat-top kernel
    return _modified_tuckey_hanning(x, 10)

def _modified_tuckey_hanning_16(x: float) -> float: # Non-Flat-top kernel
    return _modified_tuckey_hanning(x, 16)

def _quadratic_spectral(x: float) -> float: # Non-Flat-top kernel
    return (3/(x**2)) * ( np.sin(x)/(x**2) - np.cos(x) ) if x >= 0 else 0.0

def _bnhls2008(x: float) -> float: # Non-Flat-top kernel
    return (1 + x) * np.exp(-x) if x >= 0 else 0.0


########################################################################################
#                                   Kernel properties                                  #
########################################################################################

@dataclass
class _KernelPropertises:
    """
    Dataclass to hold properties of a kernel used in realized variance estimation.
    This dataclass is particularly useful for computing c* and so the optimal bandwidth for the kernel.
    If the kernel is not smooth, kappa__0 must be provided. Otherworwise, it will be computed using 
    finite difference approximation.
    We handled most of the kernels but this approximation could be useful for quickly
    adding new kernels.

    Attributes
    ----------
    func : Callable[..., float]
        The kernel function to be used.
    kappa__0 : float
        The first derivative of the kernel at zero, used for smoothness.
    kappa00 : float
        The second derivative of the kernel at zero, used for scaling.
    is_flat_top : bool
        Indicates if the kernel has a flat top (i.e., constant value over a range).
    kernel_type : Literal['smooth', 'kinked', 'discontinuous']
        Type of the kernel, indicating its continuity and differentiability properties, according to Barndorff-Nielsen et al. (2010).
    """
    func: Callable[..., float]
    kappa__0: float
    kappa00: float
    is_flat_top: bool # According to the Barndorff-Nielsen et al (2006) definition
    kernel_type: Literal['smooth', 'kinked', 'discontinuous']

    def _compute_c_star(self) -> float:
        if self.kernel_type == 'smooth':
            return ( (self.kappa__0 ** 2) / self.kappa00 ) ** (1/5)
        elif self.kernel_type == 'kinked':
            k_0 = first_derivative(f=self.func, x=0.0, h=1e-6, method='central')
            k_1 = first_derivative(f=self.func, x=1.0, h=1e-6, method='central')
            return ( 2 * (k_0**2 + k_1**2) / self.kappa00 ) ** (1/3)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}. Supported types are 'smooth' and 'kinked'.")

    def __post_init__(self):
        if self.kappa__0 is None:
            warnings.warn(f"If Kernel {self.func.__name__} is not smooth or continuous, kappa__0 approximation might be inaccurate.", UserWarning)
            self.kappa__0 = second_derivative(
                f=self.func,
                x=0.0,
                h=1e-6,
                method='central'
            )
        
        if self.kappa00 is None:
            warnings.warn(f"If Kernel {self.func.__name__} is not smooth or continuous, kappa00 approximation mignt be inaccurate.", UserWarning)
            self.kappa00 = numerical_integral(
                func=lambda x: self.func(x)**2,
                a=0.0,
                b=1.0,
                num_points=10001,
                method='simpson'
            )
        
        self.c_star: float = self._compute_c_star()

# TODO: check moments for Tukey-Hanning kernels by calculating derivatives for them
_PROPERTIES = {
    'bartlett': _KernelPropertises(func=_bartlett, kappa__0=0, kappa00=1/3, is_flat_top=True, kernel_type='kinked'),
    'epanechnikov': _KernelPropertises(func=_epanechnikov, kappa__0=2, kappa00=8/15, is_flat_top=True, kernel_type='kinked'),
    'second-order': _KernelPropertises(func=_second_order, kappa__0=2, kappa00=1/5, is_flat_top=True, kernel_type='kinked'),
    'parzen': _KernelPropertises(func=_parzen, kappa__0=12, kappa00=0.269, is_flat_top=True, kernel_type='smooth'),
    'tuckey-hanning': _KernelPropertises(func=_tuckey_hanning, kappa__0=None, kappa00=0.375, is_flat_top=True, kernel_type='smooth'),
    'm-tuckey-hanning-2': _KernelPropertises(func=_modified_tuckey_hanning_2, kappa__0=None, kappa00=0.218, is_flat_top=True, kernel_type='smooth'),
    'm-tuckey-hanning-5': _KernelPropertises(func=_modified_tuckey_hanning_5, kappa__0=None, kappa00=0.097, is_flat_top=True, kernel_type='smooth'),
    'm-tuckey-hanning-10': _KernelPropertises(func=_modified_tuckey_hanning_10, kappa__0=None, kappa00=0.05, is_flat_top=True, kernel_type='smooth'),
    'm-tuckey-hanning-16': _KernelPropertises(func=_modified_tuckey_hanning_16, kappa__0=None, kappa00=0.032, is_flat_top=True, kernel_type='smooth'),
    'cubic': _KernelPropertises(func=_cubic, kappa__0=6, kappa00=0.371, is_flat_top=True, kernel_type='smooth'),
    'bnhls2008': _KernelPropertises(func=_bnhls2008, kappa__0=1, kappa00=5/4, is_flat_top=False, kernel_type='discontinuous'),
    'quadratic-spectral': _KernelPropertises(func=_quadratic_spectral, kappa__0=1/5, kappa00=3*np.pi/5, is_flat_top=False, kernel_type='discontinuous'),
}

_KERNELS = _PROPERTIES.keys()


########################################################################################
#                               Realized Kernel Computing                              #
########################################################################################

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
    return _PROPERTIES[kernel].c_star if kernel in _PROPERTIES else None

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
    elif isinstance(bandwidth, int) and bandwidth < 1 or bandwidth > n:
        raise ValueError("Bandwidth must be a positive integer less than or equal to the number of returns.")
    H = int(bandwidth)
    
    if kernel not in _KERNELS:
        raise ValueError(f"Invalid kernel type. Supported kernels are: {', '.join(_KERNELS)}")
    k = _PROPERTIES[kernel].func

    hs = np.arange(-H, H + 1)
    # weights = np.array([k(h / (H + 1)) for h in hs])
    weights = np.array([k((h - 1) / H) for h in hs])
    gamma_h = np.array([
        np.sum([returns[j] * returns[j-abs(h)] for j in range(abs(h) + 1, n)]) for h in hs
    ])

    if dof_adjustment:
        adj_factors = n / (n - np.abs(hs))
    else:
        adj_factors = np.ones(2 * H + 1)

    rk = np.sum(weights * gamma_h * adj_factors)

    # gamma_pos = np.array([np.dot(returns[:n - h], returns[h:]) for h in range(H + 1)])
    # gamma_neg = np.array([np.dot(returns[h:], returns[:n - h]) for h in range(H + 1)])

    # weights = np.empty(H + 1)
    # weights[0] = 1.0
    # weights[1:] = np.array([k((h - 1) / H) for h in range(1, H + 1)])
    # # weights = np.array([k(h / (H + 1)) for h in range(H)])

    # if dof_adjustment:
    #     adj_factors = n / (n - np.arange(H + 1))
    # else:
    #     adj_factors = np.ones(H + 1)

    # rk = np.sum(weights * adj_factors * (
    #     gamma_pos + np.where(np.arange(H + 1) == 0, 0.0, gamma_neg)
    # ))

    return rk