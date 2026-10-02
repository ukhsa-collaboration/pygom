from sympy.core.function import diff
from sympy import Matrix

from .mathsmethod import NumericMethod
from .._model_verification import simplifyEquation

# class GradJacobian(NumericMethod):
#     method_name = 'grad_jacobian'

#     depends_on = ['grad']

#     def get_equation(self):
#         '''
#         Return the jacobian of the gradient in algebraic form

#         Returns
#         -------
#         :class:`sympy.matrices.matrices`
#             A matrix of dimension [number of state *
#             number of parameters x number of state]

#         See also
#         --------
#         :meth:`.get_grad_eqn`

#         '''
#         self._GradJacobian = sympy.zeros(
#             self._model_spec.num_state * self._model_spec.num_param,
#             self._model_spec.num_state
#         )
        
#         G = self._method_register['grad'].get_equation()

#         for param_k in range(self._model_spec.num_param):
#             for state_i in range(self._model_spec.num_state):
#                 for state_j, state in enumerate(self._model_spec.state_list):
#                     z = param_k * self._model_spec.num_state + state_i
#                     self._GradJacobian[z, state_j] = diff(G[state_i, param_k], state, 1)
#         # end of the triple loop.  All elements are now filled

#         return self._GradJacobian


class ODEMixedHessianStatesParams(NumericMethod):
    method_name = "ode_mixed_hessian_states_params"

    depends_on = ["ode"]

    def get_equation(self):
        '''
        Return the mixed Hessian of the ODE system with respect
        to states and parameters.

        Returns
        -------
        list
            List of length num_state.

            Entry i is a matrix of dimension
            [number of states x number of parameters]

            whose (j, k) element is

                d²f_i / (dx_j dp_k)

        Notes
        -----
        We deliberately return a list instead of a 3D tensor
        to avoid ambiguity regarding axis ordering.
        '''

        states = self._model_spec.state_list
        params = self._model_spec.param_list

        H = []

        for eqn in self._method_register["ode"].get_equation():
            H_i = Matrix([
                [
                    eqn.diff(state).diff(param)
                    for param in params
                ]
                for state in states
            ])

            H.append(H_i)

        return H


class EventRatesMixedHessianStatesParams(NumericMethod):
    method_name = "event_rates_mixed_hessian_states_params"

    depends_on = ["event_rate_vector"]

    def get_equation(self):
        '''
        Return the mixed Hessian of the Event Rate system with respect
        to states and parameters.

        Returns
        -------
        list
            List of length num_state.

            Entry i is a matrix of dimension
            [number of states x number of parameters]

            whose (j, k) element is

                d²f_i / (dx_j dp_k)

        Notes
        -----
        We deliberately return a list instead of a 3D tensor
        to avoid ambiguity regarding axis ordering.
        '''

        states = self._model_spec.state_list
        params = self._model_spec.param_list

        H = []

        for eqn in self._method_register["event_rate_vector"].get_equation():
            H_i = Matrix([
                [
                    eqn.diff(state).diff(param)
                    for param in params
                ]
                for state in states
            ])

            H.append(H_i)

        return H