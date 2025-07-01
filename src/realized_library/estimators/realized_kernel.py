import warnings
from dataclasses import dataclass
import numpy as np
from typing import Literal, Union, Callable
from realized_library.utils.derivative_approximation import first_derivative, second_derivative, third_derivative, fourth_derivative
from realized_library.utils.integral_approximation import numerical_integral

# Ressources:
# - http://dx.doi.org/10.2139/ssrn.620203
# - https://github.com/jonathancornelissen/highfrequency
# - https://web.archive.org/web/20220903214102/https://realized.oxford-man.ox.ac.uk/documentation/econometric-methods


########################################################################################
#                                   Kernel definitions                                 #
########################################################################################

def _bartlett(x: float) -> float: # Flat-top kernel
    return 1 - x

def _epanechnikov(x: float) -> float: # Flat-top kernel
    return 1 - x**2

def _second_order(x: float) -> float: # Flat-top kernel
    return 1 - 2 * x + x**2

def _cubic(x: float) -> float: # Non-Flat-top kernel
    return 1 - 3 * x**2 + 2 * x**3

def _fifth_order(x: float) -> float: # Non-Flat-top kernel
    return 1 - 10 * x**3 + 15 * x**4 - 6 * x**5

def _sixth_order(x: float) -> float: # Non-Flat-top kernel
    return 1 - 15 * x**4 + 24 * x**5 - 10 * x**6

def _seventh_order(x: float) -> float: # Non-Flat-top kernel
    return 1 - 21 * x**5 + 35 * x**6 - 15 * x**7

def _eighth_order(x: float) -> float: # Non-Flat-top kernel
    return 1 - 28 * x**6 + 48 * x**7 - 21 * x**8

def _parzen(x: float) -> float: # Non-Flat-top kernel
    return (1 - 6 * x**2 + 6 * x**3) * float(0 <= x <= 0.5) + (2 * (1 - x)**3) * float(0.5 < x < 1)
    
def _tuckey_hanning(x: float) -> float: # Non-Flat-top kernel
    return 0.5 * (1 + np.cos(np.pi * x))

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
    return (3/(x**2)) * ( np.sin(x)/x - np.cos(x) )

def _fejer(x: float) -> float: # Non-Flat-top kernel
    return ( np.sin(x) / x ) ** 2

def _modified_tuckey_hanning_inf(x: float) -> float: # Non-Flat-top kernel
    return np.sin( np.exp(-x) * np.pi/2 ) ** 2

def _bnhls(x: float) -> float: # Non-Flat-top kernel
    return (1 + x) * np.exp(-x)


########################################################################################
#                                   Kernel properties                                  #
########################################################################################

@dataclass
class _KernelPropertises:
    """
    Dataclass to hold properties of a kernel used in realized variance estimation.
    This dataclass is particularly useful for computing c* and so the optimal bandwidth for the kernel.
    If the kernel is not defined in the class, it will compute an approximation for c_star.

    Attributes
    ----------
    name : str
        Name of the kernel.
    func : Callable[..., float]
        The kernel function to be used.
    type : Literal[1, 2, 3]
        Type of kernel:
        - 1: Flat-top-smooth kernel with n^(1/4) rate
        - 2: Flat-top kinked kernel with n^(1/6) rate
        - 3: Non-flat-top smooth kernel with n^(1/5) rate
    """
    name: str 
    k: Callable[..., float]
    type: Literal[1, 2, 3]

    def _compute_c_star(self) -> float:
        if self.name == 'bartlett' or self.name == 'twoscale':
            return 2.28
        elif self.name == 'second-order':
            return 3.42
        elif self.name == 'epanechnikov':
            return 2.46
        elif self.name == 'cubic':
            return 3.68
        elif self.name == '5thorder':
            return 3.70
        elif self.name == '6thorder':
            return 3.97
        elif self.name == '7thorder':
            return 4.11
        elif self.name == '8thorder':
            return 4.31
        elif self.name == 'parzen':
            return 4.77
        elif self.name == 'tuckey-hanning':
            return 3.70
        elif self.name == 'm-tuckey-hanning-2':
            return 5.74
        elif self.name == 'm-tuckey-hanning-5':
            return 8.07
        elif self.name == 'm-tuckey-hanning-10':
            return 24.79
        elif self.name == 'm-tuckey-hanning-16':
            return 39.16
        elif self.name == 'non-flat-parzen':
            return (12**2 / 0.269) ** (1/5)
        elif self.name == 'quadratic-spectral':
            return ((1/5)**2 / (3 * np.pi / 5)) ** (1/5)
        elif self.name == 'fejer':
            return ((2/3)**2 / (np.pi / 3)) ** (1/5)
        elif self.name == 'thinf':
            return ((np.pi**2 / 2)**2 / 0.52) ** (1/5)
        elif self.name == 'bnhls':
            return (1**2 / (5/4)) ** (1/5)
        else:
            warnings.warn(f"Kernel '{self.name}' does not have a predefined c_star value. Using approximation.")
            return self._approximate_c_star()

    def _approximate_c_star(self) -> float:
        """
        Approximate the c_star value for the kernel using numerical integration and derivatives.

        Returns
        -------
        float
            Approximated c_star value for the kernel.
        Raises
        ------
        ValueError
            If the kernel type is not supported or if the kernel function is not defined.
        """
        if self.type == 1:            
            k_00 = numerical_integral(
                func=lambda x: self.k(x)**2, 
                a=0.0, b=1.0, num_points=1001, method='simpson'
            )
            k_02 = numerical_integral(
                func=lambda x: self.k(x) * second_derivative(f=self.k, x=x, h=1e-3, method='central'),
                a=0.0, b=1.0, num_points=1001, method='simpson'
            )
            k___0 = third_derivative(f=self.k, x=0.0, h=1e-3, method='forward')
            k04 = numerical_integral(
                func=lambda x: self.k(x) * fourth_derivative(f=self.k, x=x, h=1e-3, method='central'),
                a=0.0, b=1.0, num_points=1001, method='simpson'
            )
            f = k___0 + k04
            return np.sqrt( (1/k_00) * ( -k_02 + np.sqrt( (k_02**2) + 3 * k_00 * f ) ) )
        elif self.type == 2:
            k_0 = first_derivative(f=self.k, x=0.0, h=1e-3, method='forward')
            k_1 = first_derivative(f=self.k, x=1.0, h=1e-3, method='backward')
            k00 = numerical_integral(func=lambda x: self.k(x)**2, a=0.0, b=1.0, num_points=1001, method='simpson')
            return ( 2 * (k_0**2 + k_1**2) / k00 ) ** (1/3)
        elif self.type == 3:
            k__0 = second_derivative(f=self.k, x=0.0, h=1e-3, method='forward')
            k00 = numerical_integral(func=lambda x: self.k(x)**2, a=0.0, b=1.0, num_points=1001, method='simpson')
            return ( (k__0 ** 2) / k00 ) ** (1/5)
        else:
            raise ValueError(f"Unsupported kernel type: {self.type}. Supported types are flat-top-smooth (1), flat-top-kinked (2), and non-flat-top (3).")

    def __post_init__(self):
        try:
            self.c_star: float = self._compute_c_star()
        except Exception as e:
            # raise ValueError(f"Error computing c_star for kernel '{self.name}': {e}")
            warnings.warn(f"Error computing c_star for kernel '{self.name}': {e}.")
            self.c_star = None
            pass
    

    def optimal_bandwidth(self, n: int, omega2_est: float, iq_est: float, c_star: float = None) -> float:
        """
        Compute the optimal bandwidth for the kernel based on the number of observations, estimated noise variance, and integrated variance.
        This method estimates the optimal bandwidth using the formula:
        .. math::
            H = c^* \cdot \left( \\frac{\\omega^2}{IV} \\right)^{\\zeta_2} \cdot n^{\\zeta_1}
        where :math:`c^*` is the optimal constant for the kernel, :math:`\\omega^2` is the estimated noise variance,
        :math:`IV` is the estimated integrated variance, and :math:`n` is the number of observations.
        Parameters
        ----------
        n : int
            Number of observations (length of the price series).
        omega2_est : float
            Estimated noise variance (ω^2).
        iq_est : float
            Estimated integrated quartticity (IQ).
        c_star : float, optional
            Predefined c_star value for the kernel. If None, it will be computed from the kernel properties.
        
        Returns
        -------
        float
            Optimal bandwidth for the specified kernel type.
        
        Raises
        ------
        ValueError
            If the kernel type is not supported or if the number of observations is less than 1.
        """
        c_star = c_star if c_star is not None else self.c_star
        if c_star is None:
            raise ValueError(f"Kernel '{self.name}' does not have a predefined c_star value. Please provide a valid c_star value.")
        
        zeta2_est = omega2_est / np.sqrt(iq_est)
        if self.type == 1:
            return self.c_star * zeta2_est**(1/2) * n**(1/2)
        elif self.type == 2:
            return self.c_star * zeta2_est**(2/3) * n**(2/3)
        elif self.type == 3:
            return self.c_star * zeta2_est**(2/5) * n**(3/5)
        
    def get_weights(self, H: int) -> np.ndarray:
        """
        Get the weights for the kernel function based on the bandwidth H.
        This method computes the weights for the kernel function at different lags.

        Parameters
        ----------
        H : int
            Bandwidth for the kernel.

        Returns
        -------
        np.ndarray
            Array of weights for the kernel function.
        """
        if self.name == 'non-flat-top-parzen':
            return np.array([self.k(h / (H+1)) for h in np.arange(1, H + 1)])
        elif self.name == 'quadratic-spectral' or self.name == 'fejer':
            return np.array([self.k(h / (H+1)) for h in np.arange(1, 30*H + 1)])
        elif self.name == 'thinf':
            return np.array([self.k(h / (H+1)) for h in np.arange(1, 4*H + 1)])
        elif self.name == 'bnhls':
            return np.array([self.k(h / (H+1)) for h in np.arange(1, 10*H + 1)])
        else:
            return np.array([self.k((h - 1) / H) for h in np.arange(1, H + 1)])

_KERNELS = {

    # Kinked flat-top kernels, rate n^(1/6)
    'bartlett': _KernelPropertises(name='bartlett', k=_bartlett, type=2),
    # 'twoscale': _KernelPropertises(name='twoscale', k=_bartlett, type=2),
    'epanechnikov': _KernelPropertises(name='epanechnikov', k=_epanechnikov, type=2),
    'second-order': _KernelPropertises(name='second-order', k=_second_order, type=2),

    # Smooth flat-top kernels, rate n^(1/4)
    'cubic': _KernelPropertises(name='cubic', k=_cubic, type=1),
    '5thorder': _KernelPropertises(name='5thorder', k=_fifth_order, type=1),
    '6thorder': _KernelPropertises(name='6thorder', k=_sixth_order, type=1),
    '7thorder': _KernelPropertises(name='7thorder', k=_seventh_order, type=1),
    '8thorder': _KernelPropertises(name='8thorder', k=_eighth_order, type=1),
    'parzen': _KernelPropertises(name='parzen', k=_parzen, type=1),
    'tuckey-hanning': _KernelPropertises(name='tuckey-hanning', k=_tuckey_hanning, type=1),
    'm-tuckey-hanning-2': _KernelPropertises(name='m-tuckey-hanning-2', k=_modified_tuckey_hanning_2, type=1),
    'm-tuckey-hanning-5': _KernelPropertises(name='m-tuckey-hanning-5', k=_modified_tuckey_hanning_5, type=1),
    'm-tuckey-hanning-10': _KernelPropertises(name='m-tuckey-hanning-10', k=_modified_tuckey_hanning_10, type=1),
    'm-tuckey-hanning-16': _KernelPropertises(name='m-tuckey-hanning-16', k=_modified_tuckey_hanning_16, type=1),
    
    # Non-flat-top kernels, rate n^(1/5)
    'non-flat-parzen': _KernelPropertises(name='parzen', k=_parzen, type=3),
    # 'quadratic-spectral': _KernelPropertises(name='quadratic-spectral', k=_quadratic_spectral, type=3),
    'fejer': _KernelPropertises(name='fejer', k=_fejer, type=3),
    'thinf': _KernelPropertises(name='thinf', k=_modified_tuckey_hanning_inf, type=3),
    'bnhls': _KernelPropertises(name='bnhls', k=_bnhls, type=3),
    
}


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
    return list(_KERNELS.keys())

def add_custom_kernel(
    name: str,
    k: Callable[..., float],
    type: Literal[1, 2, 3],
    c_star: Union[float, None] = None,
) -> None:
    """
    Add a new kernel to the list of supported kernels.
    This addition will not be persistent across sessions.

    Parameters
    ----------
    name : str
        Name of the kernel.
    k : Callable[..., float]
        The kernel function to be used.
    type : Literal[1, 2, 3]
        Type of kernel:
        - 1: Flat-top-smooth kernel with n^(1/4) rate
        - 2: Flat-top kinked kernel with n^(1/6) rate
        - 3: Non-flat-top smooth kernel with n^(1/5) rate
    c_star : float, optional
        Predefined c_star value for the kernel. If None, it will be computed.
    
    Raises
    ------
    ValueError
        If the kernel name already exists or if the type is not supported.
    """
    if name in _KERNELS:
        raise ValueError(f"Kernel '{name}' already exists.")
    if type not in [1, 2, 3]:
        raise ValueError("Kernel type must be one of: 1 (flat-top-smooth), 2 (flat-top-kinked), or 3 (non-flat-top).")
    
    _KERNELS[name] = _KernelPropertises(name=name, k=k, type=type)
    if c_star is not None:
        _KERNELS[name].c_star = c_star

def get_c_star(
    kernel: str,
) -> float:
    if kernel not in _KERNELS:
        raise ValueError(f"Invalid kernel type. Supported kernels are: {', '.join(_KERNELS)}")
    return _KERNELS[kernel].c_star

def optimal_bandwidth(
    kernel: str,
    n: int,
    omega2_est: float,
    iq_est: float,
    c_star: Union[float, None] = None
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
    iq_est : float
        Estimated integrated quartticity (IQ).
    c_star : float, optional
        Predefined c_star value for the kernel. If None, it will be computed from the kernel properties.

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
    
    return _KERNELS[kernel].optimal_bandwidth(n, omega2_est, iq_est, c_star=c_star)

def compute(
    prices: np.ndarray,
    bandwidth: int,
    kernel: str,
    dof_adjustment: bool = True
) -> float:
    """
    Compute the realized variance (sum of squared log returns) from high-frequency prices.

    Parameters
    ----------
    prices : np.ndarray
        Array of strictly positive price observations.
    bandwidth : int
        Bandwidth for the kernel estimator.
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
        raise ValueError("Bandwidth must be specified. Use 'opt' for optimal bandwidth or an integer value.")
    elif bandwidth < 1 or bandwidth > n:
        raise ValueError("Bandwidth must be a positive integer less than or equal to the number of returns.")
    H = int(bandwidth)
    
    if kernel not in _KERNELS:
        raise ValueError(f"Invalid kernel type. Supported kernels are: {', '.join(_KERNELS)}")
    kernel = _KERNELS[kernel]
    weights = kernel.get_weights(H)
    actual_H = min(H, n - 1)  # Ensure H does not exceed the number of returns

    # gamma_h = np.array([
    #     np.sum([returns[j] * returns[j-abs(h)] for j in range(abs(h) + 1, n)]) for h in hs
    # ])
    # if dof_adjustment:
    #     adj_factors = n / (n - np.abs(hs))
    # else:
    #     adj_factors = np.ones(2 * H + 1)
    # rk = np.sum(weights * gamma_h * adj_factors)
    
    # gamma_pos = np.array([np.dot(returns[:n - h], returns[h:]) for h in range(actual_H + 1)])
    # gamma_neg = np.array([np.dot(returns[h:], returns[:n - h]) for h in range(actual_H + 1)])
    # if dof_adjustment:
    #     adj_factors = n / (n - np.arange(H + 1))
    # else:
    #     adj_factors = np.ones(H + 1)
    # rk = np.sum(weights * adj_factors * ( gamma_pos + np.where(np.arange(actual_H + 1) == 0, 0.0, gamma_neg) ))

    returns_base = returns[actual_H + 1 : n - actual_H]
    gamma_minus = np.zeros(actual_H)
    gamma_plus = np.zeros(actual_H)
    for idx, h in enumerate(range(1, actual_H + 1)):
        returns_minus = returns[actual_H + 1 - h : n - actual_H - h]
        returns_plus = returns[actual_H + 1 + h : n - actual_H + h]
        gamma_minus[idx] = np.dot(returns_minus, returns_base)
        gamma_plus[idx] = np.dot(returns_base, returns_plus)
    gamma_0 = np.dot(returns_base, returns_base)
    
    if dof_adjustment:
        adj_factors = n / (n - np.arange(actual_H))
    else:
        adj_factors = np.ones(actual_H)

    rk = gamma_0 + np.dot(weights * adj_factors, (gamma_minus + gamma_plus))

    return rk