from sympy import hessian

from .mathsmethod import NumericMethod
from .._model_verification import simplifyEquation

from sympy import Array

class ODEHessianParams(NumericMethod):
    method_name = 'ode_hessian_params'
    depends_on = ['ode']

    def _build_expression(self):
        '''
        Return the Hessian of the ode in algebraic form

        Returns
        -------
        list
            list of dimension number of state, each with matrix
            [number of parameters x number of parameters] in
            :mod:`sympy.matricies.matricies`

        Notes
        -----
        We deliberately return a list instead of a 3d array of a
        tensor to avoid confusion

        '''

        params = self._model_spec.parameter_symbol_list

        ode = self._method_register["ode"].get_equation(expand_derived_params=True)

        H = Array([
            hessian(eqn, params).tolist()
            for eqn in ode
        ])

        H = self._collapse_derived_parameters(H)

        self._sympy_expression =  H

class ODEHessianStates(NumericMethod):
    method_name = 'ode_hessian_states'
    depends_on = ['ode']

    def _build_expression(self):
        '''

        '''

        states = self._model_spec.state_symbol_list

        ode = self._method_register["ode"].get_equation(expand_derived_params=True)

        H = Array([
            hessian(eqn, states).tolist()
            for eqn in ode
        ])

        H = self._collapse_derived_parameters(H)

        self._sympy_expression =  H

class EventRatesHessianParams(NumericMethod):
    method_name = 'event_rates_hessian_params'
    depends_on = ['event_rate_vector']

    def _build_expression(self):
        '''

        '''

        params = self._model_spec.parameter_symbol_list

        erv = self._method_register["event_rate_vector"].get_equation(expand_derived_params=True)

        H = Array([
            hessian(eqn, params).tolist()
            for eqn in erv
        ])

        H = self._collapse_derived_parameters(H)

        self._sympy_expression =  H

class EventRatesHessianStates(NumericMethod):
    method_name = 'event_rates_hessian_states'
    depends_on = ['event_rate_vector']

    def _build_expression(self):
        '''

        '''

        states = self._model_spec.state_symbol_list

        erv = self._method_register["event_rate_vector"].get_equation(expand_derived_params=True)

        H = Array([
            hessian(eqn, states).tolist()
            for eqn in erv
        ])

        H = self._collapse_derived_parameters(H)

        self._sympy_expression =  H
