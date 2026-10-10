"""
Stochastic stepper base class
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TimeStep:
    y_new: np.ndarray | None       # new state after timestep
    t_new: float | None            # new time after timestep
    event_counts: np.ndarray | None       # number of times each jump occured between old and new timestep
    end_sim: bool                  # True if simulation is to be prematurely ended (e.g if rates
                                   # are all = 0 and user has decided these are grounds to abort)

@dataclass
class EventStep:
    y_new: np.ndarray | None       # new state after timestep
    t_new: float | None            # new time after timestep
    event_idx: int | None          # index of jump which has occured
    end_sim: bool                  # True if simulation is to be prematurely ended (e.g if rates
                                   # are all = 0 and user has decided these are grounds to abort)

class StochasticLeap(ABC):
    """
    Base class for stocahstic stepper algorithms

    Parameters
    ----------
    event_rates : callable
        Function of time and state, f(t, y), which returns an array of the
        rates of occurance of each event.
    stoichiometry_matrix: numpy.ndarray
        Integer-valued matrix where element [i, j] represents the change in state i
        resulting from an occurance of event j.
        Note: May also be referred to as the "state change matrix" or "reaction matrix".
    y_min: numpy.ndarray
        Minimum allowed state values
    y_max: numpy.ndarray
        Maximum allowed state values
    proceed_if_rates_zero : bool
        If True, continue with simulation when reaction rates are all zero. Otherwise terminate.
    seed : int
        Seed used to initialise random number generator
    """

    def __init__(self, event_rates, stoichiometry_matrix, y_min, y_max, proceed_if_rates_zero, rng=None):
        self.event_rates = event_rates
        self.stoichiometry_matrix = stoichiometry_matrix
        self.proceed_if_rates_zero = proceed_if_rates_zero
        self.y_min = y_min
        self.y_max = y_max
        self.rng = rng or np.random.default_rng()

        n_event, n_state = stoichiometry_matrix.shape
        self.n_event = n_event
        self.n_state = n_state

    def _compute_rates_and_changes(self, t, y):
        """
        For the current timestep, calculate reaction rates and state change matrix.

        Parameters
        ----------
        t : float
            Current time.
        y : numpy.ndarray
            Current state vector.

        Returns
        -------
        rates : numpy.ndarray
            reaction rates
        changes : numpy.ndarray
            state change matrix
        """
        rates = self.event_rates(t, y)

        if np.any(rates < 0):
            raise RuntimeError(f"Negative reaction rates encountered.\nTime: {t}\nState: {y}\nRates: {rates}")
    
        changes = self.stoichiometry_matrix

        return rates, changes

    def _jump_thresholds(self, y, changes):
        """
        Calculate the maximum number of times each individual reaction can occur
        before an illegal jump is made.

        Parameters
        ----------
        y : numpy.ndarray
            Current state vector.
        changes : numpy.ndarray
            State-change matrix specifying how each reaction modifies the state.

        Returns
        -------
        numpy.ndarray
            The maximum number of times each reaction can occur without violating
            state constraints.
        """

        # Difference between current state and state limits
        min_margin = y - self.y_min
        max_margin = self.y_max - y

        # Margins need to be broadcast across all reactions
        min_margin = min_margin[:, None]
        max_margin = max_margin[:, None]

        thresholds_min = np.full_like(changes, np.inf, dtype=float)
        thresholds_max = np.full_like(changes, np.inf, dtype=float)

        np.divide(
            min_margin,
            -changes,
            out=thresholds_min,
            where=changes < 0
        )

        np.divide(
            max_margin,
            changes,
            out=thresholds_max,
            where=changes > 0
        )

        thresholds_min = np.floor(thresholds_min)
        thresholds_max = np.floor(thresholds_max)

        return np.min(np.minimum(thresholds_min, thresholds_max), axis=0)

    @abstractmethod
    def _propose_jump(self, t, y, rates, *args):
        """
        Return (jumps, dt).
        """
        pass

    @abstractmethod
    def take_step(self, t, y):
        """
        Move system one timestep from (y, t) to (y_new, t_new)
        """
        pass