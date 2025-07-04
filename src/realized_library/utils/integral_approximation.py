import numpy as np
from scipy.integrate import simps, trapz
from typing import Callable, Literal

def compute(
    func: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    num_points: int = 10001,
    method: Literal["simpson", "trapezoidal"] = "simpson"
) -> float:
    """
    Numerically integrate a function over a given interval.

    Parameters
    ----------
    func : Callable[[np.ndarray], np.ndarray]
        The function to integrate. Must accept a numpy array and return a numpy array (vectorized).
    a : float
        Start of the integration interval.
    b : float
        End of the integration interval.
    num_points : int, optional
        Number of points for discretization grid. Must be odd for Simpson's rule.
    method : {"simpson", "trapezoidal"}, optional
        Integration method to use. "simpson" (default) or "trapezoidal".

    Returns
    -------
    float
        The numerical integral over the interval [a, b].

    Raises
    ------
    ValueError
        If an unknown integration method is specified or if `num_points` is even when using Simpson
    """
    if method == "simpson" and num_points % 2 == 0:
        num_points += 1

    x = np.linspace(a, b, num_points)
    y = np.array([func(xi) for xi in x])

    if method == "simpson":
        return simps(y, x)
    elif method == "trapezoidal":
        return trapz(y, x)
    else:
        raise ValueError(f"Unknown integration method: {method}")
