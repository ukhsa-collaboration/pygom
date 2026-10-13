from .mathsmethod import NumericMethod

class ODEJacobianStates(NumericMethod):
    method_name = 'ode_jacobian_states'

    depends_on = ['ode']

    def _build_expression(self):
        '''
        Returns the jacobian of ODEs vs states in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of state]

        '''

        states = self._model_spec.state_symbol_list

        # If we leave derived params in as they are, sympy thinks they are
        # parameters and won't apply the chain rule
        ode = self._method_register["ode"].get_equation(expand_derived_params=True)

        J = ode.jacobian(states)

        # Now we have to sub the derived params back in, because the build_expression expects it

        J = self._collapse_derived_parameters(J)

        self._sympy_expression =  J

class ODEJacobianParams(NumericMethod):
    method_name = 'ode_jacobian_params'

    depends_on = ['ode']

    def _build_expression(self):
        '''

        '''

        params = self._model_spec.parameter_symbol_list

        ode = self._method_register["ode"].get_equation(expand_derived_params=True)

        J = ode.jacobian(params)

        J = self._collapse_derived_parameters(J)

        self._sympy_expression =  J


class EventRatesJacobianStates(NumericMethod):
    method_name = 'event_rates_jacobian_states'

    depends_on = ['event_rate_vector']

    def _build_expression(self):
        '''

        '''

        states = self._model_spec.state_symbol_list

        erv = self._method_register["event_rate_vector"].get_equation(expand_derived_params=True)

        J = erv.jacobian(states)

        J = self._collapse_derived_parameters(J)

        self._sympy_expression =  J


class EventRatesJacobianParams(NumericMethod):
    method_name = 'event_rates_jacobian_params'

    depends_on = ['event_rate_vector']

    def _build_expression(self):
        '''

        '''

        params = self._model_spec.parameter_symbol_list

        erv = self._method_register["event_rate_vector"].get_equation(expand_derived_params=True)

        J = erv.jacobian(params)

        J = self._collapse_derived_parameters(J)

        self._sympy_expression =  J