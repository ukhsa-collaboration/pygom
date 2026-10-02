import copy

import sympy
from sympy.core.function import diff
from ..transition import TransitionType

from .mathsmethod import NumericMethod
from .._model_verification import simplifyEquation, checkEquation

class StateChangeMatrix(NumericMethod):
    method_name = 'state_change_matrix'

    def get_equation(self):
        """
        The state change matrix, where
        .. math::
            v_{i,j} = change in state i if transition j occurs
            (in symbolic form )
        """
       
        # container for output
        vMat = sympy.zeros(self._model_spec.num_state, 
                           self._model_spec.num_event)

        for event_index, event in enumerate(self._model_spec.event_list):
            for transition in event.transition_list:
                magnitude = transition.magnitude_expression
                if transition.transition_type==TransitionType.B:
                    destination_index = self._model_spec.get_state_index(transition.destination)
                    vMat[destination_index, event_index] += magnitude
                elif transition.transition_type==TransitionType.D:
                    origin_index=self._model_spec.get_state_index(transition.origin)
                    vMat[origin_index, event_index] -= magnitude
                elif transition.transition_type==TransitionType.T:
                    origin_index=self._model_spec.get_state_index(transition.origin)
                    destination_index=self._model_spec.get_state_index(transition.destination)
                    vMat[origin_index, event_index] -= magnitude
                    vMat[destination_index, event_index] += magnitude
            
        return vMat