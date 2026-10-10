import sympy

from .mathsmethod import NumericMethod
from .._model_verification import checkEquation

class PureOdeVector(NumericMethod):
    method_name = 'pure_ode_vector'
    def get_equation(self):
        '''
        non transition terms
        '''

        pure_ode = sympy.zeros(self._parent_ode.num_state, 1)
        # Now extract any ODE contributions from ODE type transitions
        for ode in self._parent_ode.ode_list:
            origin_index=self._parent_ode._state_store.get_index(ode.origin)
            pure_ode[origin_index] += checkEquation(ode.equation, self._parent_ode)

        self._pureOdeVector=pure_ode

        return self._pureOdeVector