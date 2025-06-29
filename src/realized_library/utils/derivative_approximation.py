import numpy as np
from typing import Callable, Any

def first_derivative(
    f: Callable[..., float],
    x: float,
    h: float = 1e-5,
    method: str = 'central',
    **f_params: Any
) -> float:
    """
    Approximate the first derivative of a function at point x.

    Parameters
    ----------
    f : Callable
        The target function f(x).
    x : float
        The point at which to approximate the derivative.
    h : float, optional
        Step size for the finite difference (default is 1e-5).
    method : str, optional
        Approximation method: 'central', 'forward', or 'backward' (default is 'central').
    **f_params : dict
        Additional keyword arguments to pass to f.

    Returns
    -------
    float
        Approximation of f'(x).
    """
    if method == 'central':
        return (f(x + h, **f_params) - f(x - h, **f_params)) / (2 * h)
    elif method == 'forward':
        return (f(x + h, **f_params) - f(x, **f_params)) / h
    elif method == 'backward':
        return (f(x, **f_params) - f(x - h, **f_params)) / h
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'central', 'forward', or 'backward'.")


def second_derivative(
    f: Callable[..., float],
    x: float,
    h: float = 1e-5,
    method: str = 'central',
    **f_params: Any
) -> float:
    """
    Approximate the second derivative of a function at point x.

    Parameters
    ----------
    f : Callable
        The target function f(x).
    x : float
        The point at which to approximate the second derivative.
    h : float, optional
        Step size for the finite difference (default is 1e-5).
    method : str, optional
        Approximation method: 'central', 'forward', or 'backward' (default is 'central').
    **f_params : dict
        Additional keyword arguments to pass to f.

    Returns
    -------
    float
        Approximation of f''(x).
    """
    if method == 'central':
        return (f(x + h, **f_params) - 2 * f(x, **f_params) + f(x - h, **f_params)) / (h ** 2)
    elif method == 'forward':
        return (f(x + 2 * h, **f_params) - 2 * f(x + h, **f_params) + f(x, **f_params)) / (h ** 2)
    elif method == 'backward':
        return (f(x, **f_params) - 2 * f(x - h, **f_params) + f(x - 2 * h, **f_params)) / (h ** 2)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'central', 'forward', or 'backward'.")
