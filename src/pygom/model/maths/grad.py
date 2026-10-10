import copy

import sympy
from sympy.core.function import diff

from .mathsmethod import NumericMethod
from .._model_verification import simplifyEquation

class Grad(NumericMethod):
    method_name = 'grad'
    def get_equation(self):
        '''
        Return the gradient of the ode in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of parameters]

        '''
        ode = self._parent_ode.ode.get_equation()
        self._Grad = sympy.zeros(self._parent_ode.num_state, self._parent_ode.num_param)

        for i in range(self._parent_ode.num_state):
            # need to adjust such that the first index is not
            # included because it corresponds to time
            for j, p in enumerate(self._parent_ode._iterParamList()):
                eqn, isDifficult = simplifyEquation(diff(ode[i], p, 1))
                self._Grad[i,j] = eqn
                self._parent_ode._isDifficult = self._parent_ode._isDifficult or isDifficult

        return self._Grad