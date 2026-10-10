"""
Exact methods
"""

from dataclasses import dataclass
import numpy as np

from ..base import StochasticLeap, EventStep
from ....data_classes import SolverDiagnostics

# No exact method diagnostic outputs
@dataclass
class DirectDiagnostics(SolverDiagnostics):
    pass

@dataclass
class FirstReactionDiagnostics(SolverDiagnostics):
    pass

class ExactLeap(StochasticLeap):
    """
    Base class for exact algorithms. Building on `StochasticLeap`.
    """
    def __init__(self, event_rates, stoichiometry_matrix, y_min, y_max, proceed_if_rates_zero, rng=None):
        super().__init__(event_rates, stoichiometry_matrix, y_min, y_max, proceed_if_rates_zero, rng)
        self.diag = DirectDiagnostics()
    
    def _get_new_y(self, y, changes, transition_id):
        """
        TODO: docstring

        Calculate the new state

        Parameters
        ----------
        y : numpy.ndarray
            Current state vector
        changes : numpy.ndarray
            State-change matrix specifying how each reaction modifies the state.
        jumps : numpt.ndarray
            Number of times each reaction occurs in the current timestep

        Returns
        -------
        numpy.ndarray
            The new state vector
        """
        return y + changes[:, transition_id]


    def _zero_rate_behavior(self, t, y):
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
        self.diag.zero_rate_termination=True
        return EventStep(y_new=None, t_new=None, event_idx=None, end_sim=True)

# ============================================================
# First Reaction Method
# ============================================================
class FirstReaction(ExactLeap):
    def __init__(self, event_rates, stoichiometry_matrix, y_min, y_max, proceed_if_rates_zero, rng=None):
        super().__init__(event_rates, stoichiometry_matrix, y_min, y_max, proceed_if_rates_zero, rng)
        # self.diag = DirectDiagnostics()
        self.diag = FirstReactionDiagnostics()

    def _propose_jump(self, rates):
        jump_times = self.rng.exponential(1.0 / rates)
        transition_idx = np.argmin(jump_times)
        dt = jump_times[transition_idx]

        return transition_idx, dt
    
    def take_step(self, t, y):
        rates, changes = self._compute_rates_and_changes(t, y)

        if np.all(rates == 0):
            return self._zero_rate_behavior(t, y)

        transition_idx, dt = self._propose_jump(rates)
        y_new = self._get_new_y(y, changes, transition_idx)
        
        return EventStep(y_new=y_new, t_new=t+dt, event_idx=transition_idx, end_sim=False)

# ============================================================
# Direct Reaction Method
# ============================================================
class DirectReaction(ExactLeap):
    def __init__(self, event_rates, stoichiometry_matrix, y_min, y_max, proceed_if_rates_zero, rng=None):
        super().__init__(event_rates, stoichiometry_matrix, y_min, y_max, proceed_if_rates_zero, rng)
        # self.diag = FirstReactionDiagnostics()
        self.diag = DirectDiagnostics()

    def _propose_jump(self, rates):
        # total_rate = rates.sum()
        # transition_idx = self.rng.choice(len(rates), p=rates / total_rate)
        # dt = self.rng.exponential(1.0 / total_rate)

        total_rate = rates.sum()
        u = self.rng.random() * total_rate
        transition_idx = np.searchsorted(np.cumsum(rates), u)
        dt = self.rng.exponential(1.0 / total_rate)

        return transition_idx, dt

    def take_step(self, t, y):
        rates, changes = self._compute_rates_and_changes(t, y)

        if np.all(rates == 0):
            return self._zero_rate_behavior(t, y)

        transition_idx, dt = self._propose_jump(rates)
        y_new = self._get_new_y(y, changes, transition_idx)
        
        return EventStep(y_new=y_new, t_new=t+dt, event_idx=transition_idx, end_sim=False)