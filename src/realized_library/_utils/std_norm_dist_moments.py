import numpy as np
from scipy.special import gamma
from scipy.stats import norm
from scipy.integrate import dblquad

def mu_x(x: float) -> float:
    """
    Compute mu_x = E(|N(0,1)|^x). See Barndorff-Nielsen and Shephard (2004)
    """
    return (2**(x*0.5)) * gamma((x + 1) * 0.5) / gamma(0.5)
    # return (2**(x*0.5)) * gamma((x + 1) * 0.5) / (np.pi**(0.5))

def mu_k_p_estimation(k: float, p: float, num_samples: int = 10**6, seed: int = 42) -> float:
    """
    Monte Carlo estimate of m_{k,p} = E[|U|^p * |U + sqrt(k-1)*V|^p] where U, V ~ N(0,1) independent standard normals.
    """
    np.random.seed(seed)
    U = np.random.randn(num_samples)
    V = np.random.randn(num_samples)
    term = np.abs(U)**p * np.abs(U + np.sqrt(k - 1) * V)**p
    return np.mean(term)

def mu_k_p(k: float, p: float) -> float:
    """
    Numerical integration code to compute m_{k,p} = E[|U|^p * |U + sqrt(k-1)V|^p] using the original double-integral definition,
    where U, V ~ N(0,1) iid standard normals.
    """

    sqrt_km1 = np.sqrt(k - 1)

    def integrand(v, u):
        return (abs(u) ** p) * abs(u + sqrt_km1 * v) ** p * norm.pdf(u) * norm.pdf(v)

    # Use reasonable integration limits for N(0,1): [-8, 8]
    val, _ = dblquad(integrand, -8, 8, lambda u: -8, lambda u: 8)

    return val

    