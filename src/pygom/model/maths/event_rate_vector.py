import sympy

from .mathsmethod import NumericMethod

# class EventRateVector(NumericMethod):
#     method_name = 'event_rate_vector'
#     def _build_expression(self):
#         """
#         Get all the transitions into a vector, arranged by state to
#         state transition then the birth death processes
#         """

#         event_rate_vector = sympy.zeros(self._parent_ode.num_events, 1)
#         # Extract all info from events
#         for i, event in enumerate(self._parent_ode.event_list):
#             event_rate_vector[i] = checkEquation(
#                 event.rate,
#                 self._parent_ode
#             )

#         return event_rate_vector


class EventRateVector(NumericMethod):
    method_name = 'event_rate_vector'

    def _build_expression(self):
        """
        Get all the transitions into a vector, arranged by state to
        state transition then the birth death processes
        """

        # Container for output
        event_rate_vector = sympy.zeros(self._model_spec.num_event, 1)

        for i, event in enumerate(self._model_spec.event_list):
            event_rate_vector[i] = event.rate_expression

        self._sympy_expression = event_rate_vector