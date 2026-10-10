"""
Classes to preemptively calculate if step is likely to be illegal and the
reduce step size as a precaution
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import poisson


class TauRefiner(ABC):
    """
    Check if jump is likely to be illegal and reduce tau until acceptable risk is reached
    """
    def __init__(self):
        self.n_refinements = 0
    @abstractmethod
    def refine_tau(self, tau, thresholds, rates):
        pass

class ProbabilisticTauPrecaution(TauRefiner):
    """
    Calculate how likely the time step is to result in an illegal step and reduce
    tau until acceptable risk is reached.

    An illegal step is defined as one in which any single reaction occurs enough
    times to produce state values outside of the allowed limits.

    Parameters
    ----------
    max_retries : int
        Maximum number of attempts to find an appropriate value of tau before
        abandoning the search.
    acceptable_prob_misstep : float
        Maximum acceptable probability of a misstep.
    factor_min : float, default = 0.1
        Minimum factor by which tau can be multiplied per iteration.
    factor_max : float, default = 0.9
        Minimum factor by which tau can be multiplied per iteration.
    """
    def __init__(self, max_retries, acceptable_prob_misstep, factor_min=0.1, factor_max=0.9):
        super().__init__()
        self.target = np.log(1 - acceptable_prob_misstep)
        self.max_retries = max_retries
        self.factor_min = factor_min
        self.factor_max = factor_max

    def _prob_illegal_jump(self, tau, thresholds, rates):
        """
        Calculate log probability that at least one illegal state will be generated
        with the proposed timestep

        Parameters
        ----------
        tau : float
            Proposed time step
        thresholds : numpy.ndarray
            Maximum number of times each reaction is allowed to occur before generating
            illegal states
        rates : numpy.ndarray
            Rate at which each reaction occurs

        Returns
        -------
        float
            Transformed log probability of illegal step
        """

        means = rates * tau                                     # expected number of events per reaction
        log_p_total = poisson.logcdf(thresholds, means).sum()   # (we use logcdf for numerical stability)

        return log_p_total                                      # bear in mind: p_illegal = exp( 1 - log_p_total )

    def refine_tau(self, tau, thresholds, rates):
        """
        Refine the proposed timestep value until it meets the desired risk tolerance
        for resulting in an illegal jump

        Parameters
        ----------
        tau : float
            Proposed timestep
        thresholds : numpy.ndarray
            Maximum number of times each reaction is allowed to occur before generating
            illegal states
        rates : numpy.ndarray
            Rate at which each reaction occurs

        Returns
        -------
        float
            Refined timestep
        """
        for _ in range(self.max_retries):
            log_p_total = self._prob_illegal_jump(tau, thresholds, rates)

            if log_p_total >= self.target:
                return tau

            # ratio < 1 when tau is too big
            factor = self.target / log_p_total

            # ensure that tau does not increase and that doesn't shrink too rapidly
            factor = np.clip(factor, self.factor_min, self.factor_max)
            tau *= factor

            self.n_refinements += 1
            
        raise RuntimeError(f"No safe tau found after {self.max_retries} attempts. Step size at error: {tau}.")

class NoTauPrecaution(TauRefiner):
    def refine_tau(self, tau, thresholds, rates):
        return tau