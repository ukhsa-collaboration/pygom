from .mathsmethod import NumericMethod

class ODEJacobianStates(NumericMethod):
    method_name = 'ode_jacobian_states'

    depends_on = ['ode']

    def get_equation(self):
        '''
        Returns the jacobian of ODEs vs states in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of state]

        '''        
        # states = [s for s in self._parent_ode._iterStateList()]

        states = self._model_spec.state_list
        self._Jacobian = self._method_register['ode'].get_equation().jacobian(states)

        return self._Jacobian

class ODEJacobianParams(NumericMethod):
    method_name = 'ode_jacobian_params'

    depends_on = ['ode']

    def get_equation(self):
        '''

        '''

        params = self._model_spec.param_list
        self._Jacobian = self._method_register['ode'].get_equation().jacobian(params)

        return self._Jacobian
    

class EventRatesJacobianStates(NumericMethod):
    method_name = 'event_rates_jacobian_states'

    depends_on = ['event_rate_vector']

    def get_equation(self):
        '''

        '''

        states = self._model_spec.state_list
        self._RatesJacobian = self._method_register['event_rate_vector'].get_equation().jacobian(states)

        return self._RatesJacobian


class EventRatesJacobianParams(NumericMethod):
    method_name = 'event_rates_jacobian_params'

    depends_on = ['event_rate_vector']

    def get_equation(self):
        '''

        '''

        params = self._model_spec.param_list
        self._RatesJacobian = self._method_register['event_rate_vector'].get_equation().jacobian(params)

        return self._RatesJacobian