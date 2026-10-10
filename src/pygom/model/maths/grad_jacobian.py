import copy

import sympy
from sympy.core.function import diff

from .mathsmethod import NumericMethod
from .._model_verification import simplifyEquation

class GradJacobian(NumericMethod):
    method_name = 'grad_jacobian'
    def get_equation(self):
        '''
        Return the jacobian of the gradient in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state *
            number of parameters x number of state]

        See also
        --------
        :meth:`.get_grad_eqn`

        '''
        self._GradJacobian = sympy.zeros(self._parent_ode.num_state*
                                         self._parent_ode.num_param,
                                         self._parent_ode.num_state)
        G = self._parent_ode.grad.get_equation()
        for k in range(0, self._parent_ode.num_param):
            for i in range(0, self._parent_ode.num_state):
                for j, s in enumerate(self._parent_ode._iterStateList()):
                    z = k*self._parent_ode.num_state + i
                    eqn, isDifficult = simplifyEquation(diff(G[i,k], s, 1))
                    self._GradJacobian[z,j] = eqn
                    self._isDifficult = self._parent_ode._isDifficult or isDifficult
        # end of the triple loop.  All elements are now filled

        return self._GradJacobian