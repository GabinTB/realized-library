import warnings
from typing import Optional, Union
import numpy as np
from pandas import to_timedelta
from realized_library.utils.subsampling import compute as subsample
from realized_library.utils.preaverage import compute as preaverage
from realized_library.estimators.noise_variance import compute as noise_variance_estimation

def compute(
    prices: list[float],
    preaveraged: bool = False,
    theta: Optional[float] = 0.5,
    q: Optional[int] = 50,
    skip: int = 0,
    timestamps: Optional[np.array] = None,
    sample_size: Optional[Union[int, str]] = None,
    offset: Optional[Union[int, str]] = None
) -> float:
    """
    Computes bipower variation (BPV), skip-k bipower variation, preaveraged bipower, and subsample verisons of these:
    - If preaverage is True, computes preaveraged bipower variation.
    - If skip is greater than 0, computes skip-k bipower variation.
    - If sample_size and offset are provided, computes subsampled bipower variation with the specified parameters.

    Parameters
    ----------
    prices : list[float]
        List of prices for which to compute the bipower variation.
    preaveraged : bool, optional
        If True, computes preaveraged bipower variation. Default is False.
    theta : Optional[float], optional
        The theta parameter for preaveraged bipower variation. Must be between 0 and 1. Default is 0.5.
    q : Optional[int], optional
        The q parameter for noise variance estimation. Default is 50.
    skip : int, optional
        The number of observations to skip when computing bipower variation. Default is 0.
    timestamps : Optional[np.array], optional
        Timestamps corresponding to the prices, used for subsampling. If provided, must match the length of prices.
    sample_size : Optional[Union[int, str]], optional
        The size of the sample to be used for subsampling. If provided, must be a multiple of offset.
    offset : Optional[Union[int, str]], optional
        The offset for subsampling. If provided, must be a multiple of sample_size.

    Returns
    -------
    float
        The computed bipower variation. If preaveraged is True, returns preaveraged bipower variation.
        If sample_size and offset are provided, returns subsampled bipower variation.
    """
    if len(prices) < 2:
        raise ValueError("At least two prices are required to compute bipower variation.")

    if sample_size is not None and offset is not None:
        if timestamps is None:
            raise ValueError("Timestamps must be provided when using sample_size and offset parameters.")
        if isinstance(sample_size, str) and isinstance(offset, str):
            sample_size_ns = int(to_timedelta(sample_size).total_seconds() * 1e9)
            offset_ns = int(to_timedelta(offset).total_seconds() * 1e9)
            if sample_size_ns % offset_ns != 0:
                raise ValueError(f"Sample size {sample_size} must be a multiple of offset {offset} to reduce computation time.")
            nb_samples = sample_size_ns // offset_ns
        elif isinstance(sample_size, int) and isinstance(offset, int):
            if sample_size % offset != 0:
                raise ValueError(f"Sample size {sample_size} must be a multiple of offset {offset} to reduce computation time.")
            nb_samples = sample_size // offset
        else:
            raise ValueError("Both sample_size and offset must be either strings or integers.")

        price_subsamples, timestamps_subsamples = subsample(
            prices=prices, 
            timestamps=timestamps, 
            sample_size=sample_size, 
            offset=offset,
            nb_samples=nb_samples
        )

        # bvs = []
        # for sample in price_subsamples:
        #     if len(sample) < 2:
        #         continue
        #     bvs.append(compute(
        #         prices=sample,
        #         preaveraged=preaveraged,
        #         theta=theta,
        #         q=q,
        #         skip=skip,
        #         timestamps=None,
        #         sample_size=None,
        #         offset=None
        #     ))
        # return np.mean(bvs)

        m = len(prices)
        bvs = np.zeros(len(price_subsamples))
        total_count = 0
        for i, sample in enumerate(price_subsamples):
            if len(sample) < 2:
                bvs[i] = np.nan
            else:
                returns = np.diff(np.log(sample))
                n = len(returns)
                if i == 0:
                    base_count = n - 1 - skip
                bvs[i] = (np.pi/2) * np.abs(np.dot(returns[:n-1-skip], np.abs(returns[1+skip:n])))
                total_count += n - 1 - skip
        bvSS = np.sum(bvs) * (base_count / total_count)
        bias_scale = m / (m - 1 - skip)
        return bias_scale * bvSS

    
    returns = np.diff(np.log(prices))
    # returns = returns[~np.isnan(returns)]
    n = len(returns)

    if preaveraged:
        if skip > 0:
            warnings.warn("The skip parameter is ignored when preaveraged is True. Preaveraged bipower variation does not support skipping.")
        if theta is None or theta <= 0 or theta > 1:
            raise ValueError("Theta must be a positive number between 0 and 1 for preaveraged bipower variation.")

        omega2_est = noise_variance_estimation(prices=prices, q=q, debiasing='oomen')
        if omega2_est < 0:
            omega2_est = noise_variance_estimation(prices=prices, q=q, debiasing='bandi-russel')
        if omega2_est < 0:
            raise ValueError("Noise variance estimation failed. The estimated noise variance is negative.")
        
        preav_returns = preaverage(returns, theta=theta)
        if len(preav_returns) < 2:
            return None  # Not enough data after preaveraging
        K = int(np.ceil(np.sqrt(n) * theta))

        g = lambda x: np.minimum(x, 1 - x)
        psi1 = K * np.sum( (g(np.arange(1, K + 1) / K) - g(np.arange(0, K) / K)) ** 2)
        psi2 = (1 / K) * np.sum(g(np.arange(1, K) / K) ** 2)
    
        mu1 = np.sqrt(2/np.pi)
        const1 = n / (n - 2 * K + 2)
        const2 = 1 / (K * psi2 * (mu1**2))
        bias = psi1 * (omega2_est**2) / (theta**2 * psi2)

        lead = preav_returns[K:]
        lag = preav_returns[:-K]

        return const1 * const2 * np.sum(np.abs(lead * lag)) - bias
    
    else:
        if skip > 0:
            if (n - 1 - skip) < 1 or (2 + skip) > n:
                raise ValueError(f"The value of skip ({skip}) is too large for the sampling method used. skip should be small relative to the number of prices ({n}) computed using the chosen sampling method.")

        bv = (np.pi/2) * np.abs(np.dot(returns[:n-1-skip], np.abs(returns[1+skip:n])))
        bias_scale =  n / (n - 1 - skip)
        return bias_scale * bv