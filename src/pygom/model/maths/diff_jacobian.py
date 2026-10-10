import copy

import sympy
from sympy.core.function import diff

from .mathsmethod import NumericMethod
from .._model_verification import simplifyEquation

class DiffJacobian(NumericMethod):
    method_name = 'diff_jacobian'
    def get_equation(self):
        '''
        Returns the jacobian differentiate w.r.t. states in algebraic form

        
        Returns
        -------
        list
            list of size (num of state,) each with
            :mod:`sympy.matrices.matrices` of dimension
            [number of state x number of state]

        '''

        # self.ode.get_equation()
        diffJac = list()

        for eqn in self._parent_ode.ode.get_equation():
            J = sympy.zeros(self._parent_ode.num_state, 
                            self._parent_ode.num_state)
            for i, si in enumerate(self._parent_ode._iterStateList()):
                diffEqn, D1 = simplifyEquation(diff(eqn, si, 1))
                for j, sj in enumerate(self._parent_ode._iterStateList()):
                    J[i,j], D2 = simplifyEquation(diff(diffEqn, sj, 1))
                    self._isDifficult = self._parent_ode._isDifficult or D1 or D2
            #binding.
            diffJac.append(J)

        # extract first matrix as base.  we have to get the first element
        # as base if we want to use the class method of the object
        diffJacMatrix = diffJac[0]
        for i in range(1, len(diffJac)):
            # sympy internal matrix joining
            diffJacMatrix = diffJacMatrix.col_join(diffJac[i])

        self._diffJacobian = copy.deepcopy(diffJacMatrix)

        return self._diffJacobian