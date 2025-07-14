from typing import Literal, Optional, Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm
from scipy.integrate import quad
from realized_library._utils.std_norm_dist_moments import mu_x
from realized_library.estimators.variance.realized_variance import compute as rv
from realized_library.estimators.variance.bipower_variation import compute as bpv
from realized_library.estimators.variance.realized_kernel import compute as rk
from realized_library.estimators.variance.multipower_variation import compute as mpv
from realized_library.estimators.variance.min_rv import compute as min_rv
from realized_library.estimators.variance.med_rv import compute as med_rv

def is_jump(
    value: float,
    alpha: float = 0.01,
) -> float:
    """
    Check if the value exceeds the threshold i.e. if the null hypothesis of no jump can be rejected,
    based on the significance level alpha.
    
    Parameters
    ----------
    value : float
        The computed jump test statistic.
    alpha : float
        The significance level for the test. Default is 0.01, which corresponds to a 99% confidence level.

    Returns
    -------
    bool
        True if the value indicates a jump, False otherwise.
    """
    return abs(value) > norm.ppf(1 - alpha)

def _kappa(lambda_):
    """
    Compute κ(λ) = ∫ x² Φ(x√λ) (Φ(x√λ) - 1) ϕ(x) dx where Φ is the cumulative distribution function
    and ϕ is the probability density function of the standard normal distribution.
    """
    def integrand(x):
        PHI: callable = lambda x: norm.cdf(x)
        phi: callable = lambda x: norm.pdf(x)
        return x**2 * PHI(x * np.sqrt(lambda_)) * ( PHI(x * np.sqrt(lambda_)) - 1 ) * phi(x)
    val, _ = quad(integrand, -np.inf, np.inf)
    return val

def _Cb(gamma):
    """Compute C_b(γ) bias term as per Equation (23)"""
    lam = gamma / (1 + gamma)
    return (1 + gamma) * np.sqrt((1 + gamma) / (1 + 3 * gamma)) + gamma * np.pi * 0.5 - 1 \
            + 2 * gamma / ( (1 + lam) * np.sqrt(2 * lam + 1) ) + 2 * gamma * np.pi * _kappa(lam)

def _Cq(gamma):
    """
    Compute C_q(γ) bias term from section 3.1
    C_q(γ) = 5.46648γ² + 4γ
    """
    return 5.46648 * (gamma**2) + 4 * gamma

def _Cx(gamma):
    """
    Compute C_x(γ) bias term as per Equation (23)
    C_x(γ) = 13.2968γ 3 + 14.4255γ 2 + 6γ
    """
    return 13.2968 * (gamma**3) + 14.4255 * (gamma**2) + 6 * gamma

def compute(
    prices: np.ndarray,
    test: Literal["difference", "logarithmic", "ratio"] = "ratio",
    p: Literal[4, 6] = 4,
    correct_noise: bool = True,
    omega2_est: Optional[float] = None,
    very_noisy_data: bool = False
) -> Union[float, np.ndarray]:
    """
    Compute the Jiang and Oomen Jump Test flags for a given series of prices.
    "Testing for Jumps When Asset Prices are Observed with Noise - A Swap Variance Approach"
        By Jiang G.J., and Oomen R.C.A. (2008).
        DOI: 10.1016/j.jeconom.2008.04.009
    
    Parameters
    ----------
    prices : np.ndarray
        1D array of prices for the day or 2D array of daily prices with shape (m, n) (m days, n data points per day).
    test : Literal["difference", "logarithmic", "ratio"], optional
        Type of test to perform. Options are:
        - "difference": Tests the difference between SwV and RV.
        - "logarithmic": Tests the logarithm of the ratio of SwV to RV.
        - "ratio": Tests the ratio of SwV to RV.
        Default is "ratio".
    p : Literal[4, 6], optional
        The order of the multipower variation to use. Options are 4 or 6.
        Default is 4.
    correct_noise : bool, optional
        If True, applies noise correction to the computed statistics.
        Default is True.
    omega2_est : Optional[float], optional
        Estimated noise variance. If None, it will be computed from the data.
        Default is None.
    very_noisy_data : bool, optional
        If True, applies a different noise correction for very noisy data.
        This is used when the data is expected to be very noisy, such as in high-frequency trading data.
        Default is False.

    Returns
    -------
    Union[float, np.ndarray]
        The JO jump test statistic for the day or an array of statistics for multiple days.

    Raises
    ------
    ValueError
        If the test parameter is not one of "difference", "logarithmic", or "ratio".
    """
    if prices.ndim > 1:
        statistics = []
        for price_series in prices:
            if len(price_series) < 2:
                raise ValueError("Each daily series must contain at least two entries.")
            statistics.append(compute(price_series, test=test, p=p, correct_noise=correct_noise, omega2_est=omega2_est, very_noisy_data=very_noisy_data))
        return np.array(statistics)

    N = len(prices)

    # Compute SwV (Swap Variance)
    simple_returns = (prices[1:] / prices[:-1]) - 1 # Ri
    log_returns = np.diff(np.log(prices)) # ri
    SwV = 2 * np.sum(simple_returns - log_returns)

    # Compute Ω_SwV
    n = len(log_returns)
    mklr_windows = sliding_window_view(log_returns, window_shape=p)
    product_terms = np.prod(np.abs(mklr_windows) ** (6/p), axis=1)
    omega_SwV = ( mu_x(6) / 9 ) * ( (n**3) * (mu_x(6/p)**(-p)) / (n - p + 1) ) * np.sum(product_terms)
    # omega_SwV = mpv.compute(prices=prices, m=p, r=6)

    # Compute RV (Realized Variance) and BPV (Bipower Variation)
    RV = rv(prices)
    BPV_star = bpv(prices) # mpv(prices, 2, 2) = dirty estimate of V(0,1) since BPV is computed on noisy data
    V_0_1 = BPV_star

    # If correct_noise is True, we apply the noise correction from proposition 3.2
    if correct_noise:
        if omega2_est is None:
            # omega2_est = RV / 2  # "[...] In finite sample this estimator [of variance noise] can be severely biased"
            omega2_est = - np.sum(log_returns[:-1] * log_returns[1:]) / (N - 1)
        
        # Robust estimator of daily return variance (e.g. 5-minute RV, RK, MPV or Min/MedRV))
        # V_bar = rk(prices, bandwidth=10, kernel="")
        # V_bar = mpv(prices, m=3, r=2)
        V_bar = min_rv(prices)
        # V_bar = med_rv(prices)


        gamma = N * omega2_est / V_bar
        BPV = BPV_star / (1 + _Cb(gamma)) # Noise correction for BPV, noted BPV* that estimates V_(0,1)
        V_0_1 = BPV + 2 * N * omega2_est # Estimation of V_(0,1)*
        
        if very_noisy_data:
            Q_0_1 = V_0_1**2 # Noise corrected estimation of Integrated Quarticity = Q_(0,1)* when data is very noisy
            X_0_1 = V_0_1**3 # Noise corrected estimation of Integrated Sixticity = X_(0,1)* when data is very noisy
        else:
            Q_0_1 = omega_SwV / (1 + _Cq(gamma)) # Noise corrected estimation of Integrated Quarticity = Q_(0,1)*
            X_0_1 = omega_SwV / (1 + _Cx(gamma)) # Noise corrected estimation of Integrated Sixticity = X_(0,1)*
        
        omega_SwV = 4 * N * (omega2_est**3) + 12 * (omega2_est**2) * BPV + 8 * omega2_est * (1/N) * Q_0_1 + (5/3) * (1/(N**2)) * X_0_1
        N = 1 # In the noise corrected case, N is not present in the test statistic so we neutralize it to 1

    if test == "difference":
        return (SwV - RV) * N / np.sqrt(omega_SwV)
    elif test == "logarithmic":
        return (np.log(SwV) - np.log(RV)) * V_0_1 * N / np.sqrt(omega_SwV)
    elif test == "ratio":
        return (1 - RV / SwV) * V_0_1 * N / np.sqrt(omega_SwV)
    else:
        raise ValueError("Test must be one of 'difference', 'logarithmic', or 'ratio'.")
