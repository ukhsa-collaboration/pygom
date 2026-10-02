import sympy
from sympy.core.function import diff

from .mathsmethod import NumericMethod

# from .._model_verification import simplifyEquation
# TODO: add back in the is difficult and simplify bits
# TODO: much repeated code below

class ODEGradParams(NumericMethod):
    method_name = 'ode_grad_params'

    depends_on = ['ode']

    def get_equation(self):
        '''
        Return the gradient of the ode wrt parameters in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of parameters]

        '''

        # container for output
        parameter_grad = sympy.zeros(
            self._model_spec.num_state,
            self._model_spec.num_param
        )

        for state_i in range(self._model_spec.num_state):
            # need to adjust such that the first index is not
            # included because it corresponds to time
            for param_j, param in enumerate(self._model_spec.param_list):
                parameter_grad[state_i, param_j] = diff(self._method_register['ode'].get_equation()[state_i], param, 1)

        return parameter_grad


class ODEGradStates(NumericMethod):
    method_name = 'ode_grad_states'

    depends_on = ['ode']

    def get_equation(self):
        '''

        '''

        # container for output
        state_grad = sympy.zeros(
            self._model_spec.num_state,
            self._model_spec.num_state
        )

        for state_i in range(self._model_spec.num_state):
            for state_j, state in enumerate(self._model_spec.state_list):
                state_grad[state_i, state_j] = diff(self._method_register['ode'].get_equation()[state_i], state, 1)

        return state_grad


class EventRatesGradParams(NumericMethod):
    method_name = 'event_rates_grad_params'

    depends_on = ['event_rate_vector']

    def get_equation(self):
        '''
        Return the gradient of the ode wrt parameters in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of parameters]

        '''

        # container for output
        parameter_grad = sympy.zeros(
            self._model_spec.num_event,
            self._model_spec.num_param
        )

        for event_i in range(self._model_spec.num_event):
            # need to adjust such that the first index is not
            # included because it corresponds to time
            for param_j, param in enumerate(self._model_spec.param_list):
                parameter_grad[event_i, param_j] = diff(self._method_register['event_rate_vector'].get_equation()[event_i], param, 1)

        return parameter_grad


class EventRatesGradStates(NumericMethod):
    method_name = 'event_rates_grad_states'

    depends_on = ['event_rate_vector']

    def get_equation(self):
        '''

        '''

        # container for output
        state_grad = sympy.zeros(
            self._model_spec.num_event,
            self._model_spec.num_state
        )

        for event_i in range(self._model_spec.num_event):
            for state_j, state in enumerate(self._model_spec.state_list):
                state_grad[event_i, state_j] = diff(self._method_register['event_rate_vector'].get_equation()[event_i], state, 1)

        return state_grad