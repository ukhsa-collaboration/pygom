import copy

import numpy as np

import sympy
from sympy.core.function import diff

from .mathsmethod import NumericMethod
from .._model_verification import simplifyEquation

class Hessian(NumericMethod):
    method_name = 'hessian'
    _Hessian = None
    def get_equation(self):
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
        # TODO: should be tied to the rest of the cache invalidation
        if self._Hessian is not None:
            return self._Hessian
        
        ode = self._parent_ode.ode.get_equation()
        self._Hessian = list()
        # roll out the equation one by one.  Each H below is a the
        # second derivative of f_{j}(x), the j^{th} ode.  Each ode
        # correspond to a state
        for eqn in ode:
            H = sympy.zeros(self._parent_ode.num_param, 
                            self._parent_ode.num_param)
            # although this can be simplified by first finding the gradient
            # it is not required so we will be slow here
            for i, pi in enumerate(self._parent_ode._iterParamList()):
                a = diff(eqn, pi, 1)
                for j, pj in enumerate(self._parent_ode._iterParamList()):
                    H[i,j], isDifficult = simplifyEquation(diff(a, pj, 1))
                    self._isDifficult = self._parent_ode._isDifficult or isDifficult
            # end of double loop.  Finished one state
            self._Hessian.append(H)

        return self._Hessian
    
    def __call__(self, state, time):
        """
        Evaluate the hessian given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        list
            list of dimension number of state, each with matrix
            [number of parameters x number of parameters] in
            :mod:`sympy.matricies.matricies`

        """
        A = self.eval_hessian(state=state, time=time)
        return [np.array(H, float) for H in A]

    def eval_hessian(self, parameters=None, time=None, state=None):
        '''
        Evaluate the hessian given parameters, state and time. An extension
        of :meth:`get_equation` but now also include the parameters.

        Parameters
        ----------
        parameters: list
            see :meth:`.parameters`
        time: double
            The current time
        state: array list
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        list
            list of dimension number of state, each with matrix
            [number of parameters x number of parameters] in
            :mod:`sympy.matricies.matricies`

        See Also
        --------
        :meth:`.grad`, :meth:`.eval_grad`

        '''
        if self._hasNewTransition:
            self.ode.get_equation()

        eval_param = list()
        # add time to the evaluated parameters
        eval_param.append((self._t, time))
        #eval_param = self._addTimeEvalParam(eval_param, time)
        # add the state values
#            def _addStateEvalParam(self, eval_param, state):
#        super(DeterministicOde, self).state = state
        if self._parent_ode._state is not None:
            eval_param += self._state

        # return eval_param
        # eval_param = self._addStateEvalParam(eval_param, state)

        if parameters is None:
            if self._HessianWithParam is None:
                self._computeHessianParam()
        # else:
        #     self.parameters = parameters

        if self._Hessian is None:
            self._computeHessianParam()

        H = list()
        for i in range(0, self.num_state):
            H = self._HessianWithParam[i].subs(eval_param)
        return H

    def _computeHessianParam(self):
        self._Hessian = self.get_hessian_eqn()

        self._HessianWithParam = copy.deepcopy(self._Hessian)
        for H in self._HessianWithParam:
            H = H.subs(self._parameters)

        return None