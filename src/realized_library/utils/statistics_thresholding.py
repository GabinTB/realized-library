import numpy as np
from scipy import stats
from typing import Union, List, Literal

def compute(
    statistics_matrix: np.ndarray,
    alpha: Literal["universal", float] = "universal",
    return_daily_flags = False
) -> Union[List[bool], List[np.ndarray]]:
    """
    Computes the jump detection flags based on a threshold derived from the number of observations.
    "Detecting spurious jumps in high frequency data"
    by Bajgrowicz, P., and Scaillet, O. (2009).
    DOI: NA
    "Jumps in high-frequency data: spurious detections, dynamics, and news"
    by by Bajgrowicz, P., Scaillet, O., and Treccani, A. (2015).
    DOI: 10.2139/ssrn.1343900

    Parameters
    ----------
    statistics_matrix : np.ndarray
        A 2D numpy array where each row represents a daily series of statistics.
    return_daily_flags : bool, optional
        If True, returns a list of boolean arrays indicating the presence of jumps for each daily series.
        If False, returns a list of integers indicating whether any jumps are present in each daily series
        (default is False).

    Returns
    -------
    list
        A list of either boolean arrays indicating the presence of jumps in each daily series or a list array of
        boolean values indicating the location of jumps in each daily series.
    """
    N = statistics_matrix.shape[0]
    if N == 0:
        raise ValueError("Input prices_matrix must not be empty.")
    
    if alpha == "universal":
        quantile = (2 * np.log(N)) ** (0.5)
    elif isinstance(alpha, (int, float)):
        if alpha <= 0:
            raise ValueError("Alpha must be a positive number.")
        quantile = stats.norm.ppf(1 - alpha)
    else:
        raise ValueError("Alpha must be either 'universal' or a positive float number.")

    if not return_daily_flags:
        return [np.max(np.abs(daily_series) > quantile) for daily_series in statistics_matrix]

    daily_flags_series = []
    for daily_statistics in statistics_matrix:
        n = len(daily_statistics)
        if n == 0:
            raise ValueError("Each daily series must not be empty.")
        
        jumps_present = np.max(np.abs(daily_statistics) > quantile)
        daily_flags = np.zeros(n, dtype=bool)
        if jumps_present:
            daily_flags = np.abs(daily_statistics) > quantile

        daily_flags_series.append(daily_flags)

    return daily_flags