import numpy as np
from scipy.special import gamma

def mu_x(x: float) -> float:
    """
    Compute mu_x = E(|N(0,1)|^x). See Barndorff-Nielsen and Shephard (2004)
    """
    return (2**(x*0.5)) * gamma((x + 1) * 0.5) / gamma(0.5)
    # return (2**(x*0.5)) * gamma((x + 1) * 0.5) / (np.pi**(0.5))
    