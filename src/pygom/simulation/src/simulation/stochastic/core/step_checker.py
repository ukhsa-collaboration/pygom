"""
Class to check if step is legal
"""

import numpy as np
from abc import ABC, abstractmethod
from .base import TimeStep

class JumpChecker(ABC):
    """
    The JumpChecker class determines if a proposed jump is valid
    """
    def __init__(self):
        self.n_illegal = 0

    def check_states(self, proposal : TimeStep, y_min, y_max):
        """
        Determine if all proposed states are within minimum and maximum limits

        Parameters
        ----------
        proposal : Step
            Proposed step data, from which we extract y_new
        y_min : numpy.ndarray
            Minimum allowable values for each state variable.
        y_max : numpy.ndarray
            Maximum allowable values for each state variable.
            
        Returns
        -------
        bool
            True if the proposed values are valid
        """

        y_new = proposal.y_new

        if np.any(y_new < y_min) or np.any(y_new > y_max):
            return False

        return True

    def check_reactions(self, proposal : TimeStep, thresholds):
        """
        Determine if all proposed jumps do not create illegal states individually

        Parameters
        ----------
        proposal : Step
            Propsed step data, from which we extract jumps
        constraints : Constraints
            Constraints, from which we require thresholds
        thresholds : numpy.ndarray
            Maximum number of times each reaction is allowed to occur before generating
            illegal states

        Returns
        -------
        bool
            True if the proposed jumps are valid
        """

        event_counts = proposal.event_counts

        if np.any(event_counts > thresholds):
            return False
        
        return True

    @abstractmethod
    def is_legal(self, proposal : TimeStep, y_min, y_max, thresholds):
        """
        True if proposed step satisfies constraints
        """
        pass

class CriticalReactionCheck(JumpChecker):
    """
    Check that each reaction doesn't occur enough times to
    create illegal states by itself.
    """

    def is_legal(self, proposal : TimeStep, y_min, y_max, thresholds):

        legal = self.check_reactions(proposal, thresholds) and self.check_states(proposal, y_min, y_max)

        if legal == False:
            self.n_illegal += 1

        return legal

class ForbiddenStateCheck(JumpChecker):
    """
    Check that each state stays within it's min and max limits
    after the net effect of all reactions.
    """

    def is_legal(self, proposal : TimeStep, y_min, y_max, thresholds):

        legal = self.check_states(proposal, y_min, y_max)

        if legal == False:
            self.n_illegal += 1
    
        return legal

class NoCheck(JumpChecker):
    """
    If we are using a method guaranteed not to generate illegal states, proceed without checks.
    """

    def is_legal(self, proposal : TimeStep, y_min, y_max, thresholds):
        return True
