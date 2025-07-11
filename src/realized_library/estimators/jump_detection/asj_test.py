from typing import Optional, Union, Literal
import numpy as np
from scipy.stats import norm
from realized_library._utils.std_norm_dist_moments import mu_x, mu_k_p
from realized_library._utils.hft_timeseries_data import get_time_delta
from realized_library.utils.resampling import compute as resample
from realized_library.estimators.variance.trucated_multipower_variation import compute as tpv
from realized_library.estimators.variance.multipower_variation import compute as mpv

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
    return abs(value) > norm.ppf(1 - alpha/2)  # Two-tailed test, so we divide alpha by 2

def compute(
    prices: np.ndarray,
    timestamps: np.ndarray,
    A_estimator: Literal["truncated", "multipower"] = "multipower",
    p: Optional[int] = 4,
    k: Optional[int] = 2,
    alpha: Optional[float] = 4*0.2,
    omega_bar: Optional[float] = 0.47,
) -> Union[float, np.ndarray]:
    """
    Compute the Aït-Sahalia and Jacod (ASJ) Jump Test flags for a given series of prices.
    "Testing for Jumps in a Discretely Observed Process"
        By Aït-Sahalia Y., and Jacod J. (2009).
        DOI: 10.1214/07-AOS568
    Many important case must be highlighted from the paper:
        - Case p > 2: B^(p, delta_n)_t -> B(p)_t
        - Case p = 2: B^(2, delta_n)_t -> [X, X]_t
        - Case p < 2: B^(p, delta_n)_t * (delta_n**(1 - p/2)) / m_p -> A(p)_t
        - X is continuous: B^(p, delta_n)_t * (delta_n**(1 - p/2)) / m_p -> A(p)_t
    Additionally, some remarks from the authors when choosing the free parameters p, k, α and omega_bar:
    - "In both Theorems 6 and 7 one has two “basic” parameters to choose, namely p > 3 and the integer k ≥ 2.
      For  the standardization, and except if we use V ̃nc,t in Theorem 6, we also need to choose α > 0 and ,
      which should be in (1/2 - 1/p, 1/2) for the first theorem, and in (0, 1/2) for the second one."
    - "About the real p > 3: the larger it is, the more the emphasis put on “large” jumps. Since those are in
      any case relatively easy to detect, or at least much easier than the small ones, it is probably wise to
      choose p “small.” However, choosing p close to 3 may induce a rather poor fit in the central limit theorem
      for a number n of observations which is not very large, since for p = 3 there is still a CLT, but with a bias.
      A good compromise seems to be p = 4, since in addition the computations are easier when p is an integer."
    - "About k: when k increases, we have to separate two points (1 and k when p = 4) which are further and further 
      apart, but on the other hand in (18) and (19) we see that the asymptotic variances are increasing with k.
      Furthermore, large values of k lead to a decrease in the effective sample size employed to estimate the numerator
      B^(p, k n)t of the test statistic, which is inefficient. So one should choose k not too big. Numerical experiments
      with k = 2, 3, 4 have been conducted below, showing no significant differences for the results of the test itself.
      More experiments should probably be conducted; however, we think that the choice k = 2 is in all cases reasonable."
    - "About α and omega_bar: from a number of numerical experiments, not reported here to save space, it seems that choosing
      close to 1/2 (as omega_bar = 0.47 or = 0.48), and α between 3 and 5 times the “average” value of σ , leads to the left-hand
      sides of (27) and (28) being very close to the right-hand sides, for relatively small values of n.
      Of course choosing α as above may seem circular because σt is unknown, and usually random and time-varying, but in
      practice one very often has a pretty good idea of the order of magnitude of σt , especially for financial data.
      Even if no a priori order of magnitude is available, it is possible to  estimate consistently the volatility of the
      continuous part of the semimartingale,  (∫0->t σ_s^2 ds)^1/2, in the presence of jumps; see the literature on disentangling
      jumps from diffusions cited in the Introduction. The multipower variations (22) do not suffer from the drawback of having
      to choose α and omega_bar a priori, but they cannot be used for Theorem 7, and when there are jumps the quality of the approximation
      in (26) strongly depends on the relative sizes of σ and of the cumulated jumps."

    Parameters
    ----------
    prices : np.ndarray
        1D array of intraday data of shape (1,n) (n data points) for the day or 2D array of daily intraday
        data with shape (m, n) (m days, n data points per day).
    timestamps : np.ndarray
        1D array of timestamps of shape (1,n) (n data points) corresponding to the intraday data, in nanoseconds 
        since epoch, or 2D array of daily timestamps with shape (m, n) (m days, n data points per day).
    A_estimator : Literal["truncated", "multipower"]
        The estimator to use for A(p). Default is "multipower".
    p : Optional[int]
        The order of the variation. Must be greater than 3. Default is 4.
    k : Optional[int]
        The number of observations to consider for the jump test. Must be greater than or equal to 2. Default is 2.
    alpha : Optional[float]
        The significance level for the test. Default is 4*0.2, which corresponds to a 99% confidence level.
    omega_bar : Optional[float]
        The omega_bar parameter for the truncated variation estimator. Default is 0.47.

    Returns
    -------
    Union[float, np.ndarray]
        The ASJ jump test statistic for the day or an array of statistics for multiple days.

    Raises
    ------
    ValueError
        If p is not provided or is less than or equal to 3, or if k is not provided or is less than 2.
        If the prices and timestamps do not have the same shape, or if the timestamps do not contain at least two entries.
        If the timestamps are not equally spaced.
        If the prices are not provided or do not contain at least two entries.
        If alpha or omega_bar is not provided when using the truncated variation estimator.
        If the A_estimator is not one of "truncated" or "multipower".
    Warning
        If the computed k is outside the suggested bounds based on the number of observations.
    """
    if p is None or p <= 3:
        raise ValueError("p must be provided and must respect p > 3.")
    if k is None or k < 2:
            raise ValueError("k must be provided and must respect k ≥ 2.")
    
    if prices.ndim > 1:
        statistics = []
        for price_series, timestamp_series in zip(prices, timestamps):
            if len(price_series) < 2 or len(timestamp_series) < 2:
                raise ValueError("Each daily series must contain at least two entries.")
            statistics.append(compute(prices=price_series, timestamps=timestamp_series, A_estimator=A_estimator, p=p, k=k, alpha=alpha, omega_bar=omega_bar))
        return np.array(statistics)

    def A_p(p_: float) -> float:
        """
        Realized truncated pth variation estimator.
        """
        if alpha is None:
            raise ValueError("Alpha must be provided for the truncated variation estimator.")
        if omega_bar is None:
            raise ValueError("Omega_bar must be provided for the truncated variation estimator.")
        return tpv(prices=prices, timestamps=timestamps, p=p_, alpha=alpha, omega=omega_bar, correct_scaling_bias=False)
    
    def A_r_q(r_: float, q_: float) -> float:
        """
        Multipower Variation estimator.
        "The multipower variations (22) do not suffer from the drawback of having to choose α and omega a priori, but they cannot be used for Theorem 7"
        """
        return mpv(prices=prices, timestamps=timestamps, m=q_, r=r_*q_, correct_scaling_bias=False)

    def B_p(X, p_: int) -> float:
        delta_X = np.diff(np.log(X)) # Log-returns
        return np.sum(np.abs(delta_X) ** p_)

    def M(p_: float, k_: float) -> float:
        return (1/(mu_x(p_)**2)) * ( (k**(p_-2)) * (1 + k_) * mu_x(2*p_) + (k**(p_-2)) * (k_ - 1) * (mu_x(p_)**2) - 2 * k_**(p_*0.5-1) * mu_k_p(k_, p_) )
    
    delta = get_time_delta(timestamps=timestamps, N=len(prices)-1)

    B_p_delta = B_p(prices, p)
    resampled_prices, _ = resample(prices=prices, timestamps=timestamps, sample_size=k)
    B_p_kdelta = B_p(resampled_prices, p)
    S = B_p_kdelta / B_p_delta

    if A_estimator == "truncated":
        V = delta * M(p_=p, k_=k) * A_p(p_=2*p) / (A_p(p_=p)**2)
    elif A_estimator == "multipower":
        # "In (31) we have chosen r = p/([p] + 1) and respectively q = 2[p] + 2 and q = [p] + 1.
        #  Any other choice with r ∈ (0, 2) and respectively q = 2p/r and q = p/r would do as well."
        r = int(p / (np.floor(p) + 1))
        q1 = int(2 * np.floor(p) + 2)
        q2 = int(np.floor(p) + 1)
        V = delta * M(p_=p, k_=k) * A_r_q(r_=r, q_=q1) / (A_r_q(r_=r, q_=q2)**2)
    else:
        raise ValueError(f"Invalid A_estimator: {A_estimator}. Choose 'truncated' or 'multipower'.")
    
    return ( k**(p*0.5 - 1) - S ) / np.sqrt(V)