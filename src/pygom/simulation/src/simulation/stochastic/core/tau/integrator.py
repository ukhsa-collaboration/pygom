"""
Tau Leap
"""

from dataclasses import dataclass
import numpy as np

from ..base import StochasticLeap, TimeStep
from ....data_classes import SolverDiagnostics
from .precaution import TauRefiner, ProbabilisticTauPrecaution
from .method import TauMethod, Adaptive
from ..step_checker import JumpChecker, CriticalReactionCheck

@dataclass
class TauLeapDiagnostics(SolverDiagnostics):
    n_failed_step: int = 0

class TauLeap(StochasticLeap):
    """
    Base class for tau leap algorithms. Building on `StochasticLeap`.
    """
    def __init__(self, event_rates, stoichiometry_matrix, y_min, y_max, proceed_if_rates_zero,
                 retry_max,
                 tau_rescale,
                 tau_method : TauMethod,
                 proposal_checker : JumpChecker,
                 tau_refiner : TauRefiner,
                 rng=None):
        super().__init__(event_rates, stoichiometry_matrix, y_min, y_max, proceed_if_rates_zero, rng)

        self.tau_method = tau_method
        self.proposal_checker = proposal_checker
        self.tau_refiner = tau_refiner
        self.retry_max = retry_max
        self.tau_rescale = tau_rescale
        self.diag = TauLeapDiagnostics()

    def _propose_jump(self, rates, tau):
        # jumps = np.random.poisson(rates * tau)
        event_counts = self.rng.poisson(rates * tau)
        return event_counts, tau
    
    def _modify_tau(self, tau):
        return tau * self.tau_rescale

    def _get_new_x(self, y, stoichiometry_matrix, event_counts):
        """
        Calculate the new state

        Parameters
        ----------
        y : numpy.ndarray
            Current state vector
        cstoichiometry_matrix : numpy.ndarray
            State-change matrix specifying how each reaction modifies the state.
        event_counts : numpt.ndarray
            Number of times each reaction occurs in the current timestep

        Returns
        -------
        numpy.ndarray
            The new state vector
        """
        
        return y + stoichiometry_matrix @ event_counts

    def _zero_rate_behavior(self, t, y, tau):
        """
        Decide what to do when reaction rates are all zero.
        If rates are guaranteed to remain at zero, we might wish to abort the simulation to save time.
        If rates might possibly become non zero later on, we continue.

        Parameters
        ----------
        t : float
            Current time.
        y : numpy.ndarray
            Current state vector.

        Returns
        -------
        Step
            Step.y_new = new state values (which are unchanged from old ones)
            Step.t_new = new time
            Step.jumps = reaction occurances
            Step.end_sim = True if all rates = 0 led to simulation termination
        """
        if self.proceed_if_rates_zero:
            return TimeStep(y_new=y, t_new=t+tau, event_counts=np.zeros(self.n_reactions), end_sim=False)
        else:
            self.diag.zero_rate_termination=True
            return TimeStep(y_new=None, t_new=None, event_counts=None, end_sim=True)

    def take_step(self, t, y):
        """
        Called by user. Returns (y_new, t_new).
        """
        rates, changes = self._compute_rates_and_changes(t, y)


        # Thresholds can be expensive to calculate, so only do if required
        thresholds = None
        if (
            isinstance(self.tau_method, Adaptive)
            or isinstance(self.tau_refiner, ProbabilisticTauPrecaution)
            or isinstance(self.proposal_checker, CriticalReactionCheck)
        ):
            thresholds = self._jump_thresholds(y, changes)

        tau, success = self.tau_method.compute_tau(t, y, rates, thresholds)

        if success==False:
            return TimeStep(y_new=None, t_new=None, event_counts=None, end_sim=True)

        tau = self.tau_refiner.refine_tau(tau, thresholds, rates)

        if np.all(rates == 0):
            return self._zero_rate_behavior(t, y, tau)

        for _ in range(self.retry_max + 1):
            # Propose jump
            event_counts, dt = self._propose_jump(rates, tau)

            # Calculate new state
            y_proposal = self._get_new_x(y, changes, event_counts)

            step_proposal = TimeStep(y_new=y_proposal, t_new=t+dt, event_counts=event_counts, end_sim=False)

            # Check if jump is legal
            if self.proposal_checker.is_legal(step_proposal, self.y_min, self.y_max, thresholds):
                return step_proposal
            else:
                self.diag.n_failed_step += 1
                tau = self._modify_tau(tau)
        raise RuntimeError(f"Forbidden values still encountered after {self.retry_max} attempts")