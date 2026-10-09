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

        states = self._model_spec.state_list

        ode = self._method_register["ode"].get_equation()

        ode = self.expand_derived_params(ode)

        J = ode.jacobian(states)

        J = self.compress_derived_params(J)

        return J

class ODEJacobianParams(NumericMethod):
    method_name = 'ode_jacobian_params'

    depends_on = ['ode']

    def get_equation(self):
        '''

        '''

        params = self._model_spec.param_list

        ode = self._method_register["ode"].get_equation()

        ode = self.expand_derived_params(ode)

        J = ode.jacobian(params)

        J = self.compress_derived_params(J)

        return J
    

class EventRatesJacobianStates(NumericMethod):
    method_name = 'event_rates_jacobian_states'

    depends_on = ['event_rate_vector']

    def get_equation(self):
        '''

        '''

        states = self._model_spec.state_list

        erv = self._method_register["event_rate_vector"].get_equation()

        erv = self.expand_derived_params(erv)

        J = erv.jacobian(states)

        J = self.compress_derived_params(J)

        return J


class EventRatesJacobianParams(NumericMethod):
    method_name = 'event_rates_jacobian_params'

    depends_on = ['event_rate_vector']

    def get_equation(self):
        '''

        '''

        params = self._model_spec.param_list

        erv = self._method_register["event_rate_vector"].get_equation()

        erv = self.expand_derived_params(erv)

        J = erv.jacobian(params)

        J = self.compress_derived_params(J)

        return J