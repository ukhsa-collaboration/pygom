import sympy

from .mathsmethod import NumericMethod
from .._model_verification import checkEquation

class EventRateVector(NumericMethod):
    method_name = 'event_rate_vector'
    def get_equation(self):
        """
        Get all the transitions into a vector, arranged by state to
        state transition then the birth death processes
        """

        event_rate_vector = sympy.zeros(self._parent_ode.num_events, 1)
        # Extract all info from events
        for i, event in enumerate(self._parent_ode.event_list):
            event_rate_vector[i]=checkEquation(event.rate, 
                                               self._parent_ode)

        return event_rate_vector