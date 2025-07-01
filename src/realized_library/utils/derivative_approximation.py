import numpy as np
from typing import Callable

def first_derivative(f: Callable[[float], float], x: float, h: float = 1e-5, method: str = 'central') -> float:
    if method == 'central':
        return (f(x + h) - f(x - h)) / (2 * h)
    elif method == 'forward':
        return (f(x + h) - f(x)) / h
    elif method == 'backward':
        return (f(x) - f(x - h)) / h
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'central', 'forward', or 'backward'.")

def second_derivative(f: Callable[[float], float], x: float, h: float = 1e-5, method: str = 'central') -> float:
    if method == 'central':
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
    elif method == 'forward':
        return (f(x + 2 * h) - 2 * f(x + h) + f(x)) / (h ** 2)
    elif method == 'backward':
        return (f(x) - 2 * f(x - h) + f(x - 2 * h)) / (h ** 2)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'central', 'forward', or 'backward'.")

def third_derivative(f: Callable[[float], float], x: float, h: float = 1e-5, method: str = 'central') -> float:
    if method == 'central':
        return (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h ** 3)
    elif method == 'forward':
        return (f(x + 3 * h) - 3 * f(x + 2 * h) + 3 * f(x + h) - f(x)) / (h ** 3)
    elif method == 'backward':
        return (f(x) - 3 * f(x - h) + 3 * f(x - 2 * h) - f(x - 3 * h)) / (h ** 3)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'central', 'forward', or 'backward'.")

def fourth_derivative(f: Callable[[float], float], x: float, h: float = 1e-5, method: str = 'central') -> float:
    if method == 'central':
        return (f(x - 2 * h) - 4 * f(x - h) + 6 * f(x) - 4 * f(x + h) + f(x + 2 * h)) / (h ** 4)
    elif method == 'forward':
        return (f(x + 4 * h) - 4 * f(x + 3 * h) + 6 * f(x + 2 * h) - 4 * f(x + h) + f(x)) / (h ** 4)
    elif method == 'backward':
        return (f(x) - 4 * f(x - h) + 6 * f(x - 2 * h) - 4 * f(x - 3 * h) + f(x - 4 * h)) / (h ** 4)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'central', 'forward', or 'backward'.")