"""
Functions which may be repeatedly used during tests.

For example, standard epidemiology results.
"""

from scipy.special import lambertw
from scipy.stats import norm
import numpy as np

def final_size_mn(R0, s0=1.0):
    """
    Compute mean epidemic final size (fraction infected)

    R0: basic reproductive number
    s0: initial fraction susceptible
    """
    if (s0>1) or (s0<0):
        raise ValueError("s0 must be between 0 and 1")

    Reff = s0*R0

    if Reff <= 1:
        return 0

    s_final = -(1/R0) * lambertw(-Reff * np.exp(-Reff) ).real

    return s0 - s_final

def final_size_sd(R0, N, r=1):
    """
    Compute standard deviation of epidemic final size, assuming
    normal distribution approximation valid (high N)

    Following eq (12) of https://doi.org/10.1016/j.mbs.2010.01.006
    (https://www.stat.berkeley.edu/~aldous/260-FMIE/Papers/britton.pdf)

    R0: basic reproductive number
    N: population size
    r: where "r**2 = V (I)/(E(I))^2 is the squared coefficient of variation of the infectious period".
       For, exponential distribution, r=1.
    """

    z = final_size_mn(R0)

    return np.sqrt(N * z * (1-z) * (1 + r**2 *(1-z) * R0)) / (1 - R0 * (1-z))