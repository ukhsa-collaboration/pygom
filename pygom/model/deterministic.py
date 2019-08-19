"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    This module is defined such that operation on ode are all gathered
    in one place.  Future extension of operations should be added here

"""

__all__ = ['DeterministicOde']

import copy
import io
from numbers import Number

# import sympy.core.numbers
import numpy as np
import sympy
import scipy.linalg

from sympy.core.function import diff

from .base_ode_model import BaseOdeModel
from ._model_errors import ArrayError, InputError, \
    IntegrationError, InitializeError
from ._model_verification import simplifyEquation

from . import ode_utils
from . import _ode_composition

class HasNewTransition(ode_utils.CompileCanary):
    states = ['ode', 'Jacobian', 'diffJacobian', 'grad', 'GradJacobian']

class DeterministicOde(BaseOdeModel):
    '''
    This contains the interface and operation
    built above the already defined set of ode

    Parameters
    ----------
    state: list
        A list of states (string)
    param: list
        A list of the parameters (string)
    derived_param: list
        A list of the derived parameters (tuple of (string,string))
    transition: list
        A list of transition (:class:`.Transition`)
    birth_death: list
        A list of birth or death process (:class:`.Transition`)
    ode: list
        A list of ode (:class:`Transition`)

    '''

    def __init__(self,
                 state=None,
                 param=None,
                 derived_param=None,
                 transition=None,
                 birth_death=None,
                 ode=None):
        '''
        Constructor that is built on top of a BaseOdeModel
        '''

        super(DeterministicOde, self).__init__(state,
                                               param,
                                               derived_param,
                                               transition,
                                               birth_death,
                                               ode)

        self._hasNewTransition = HasNewTransition()
        self._ode = None
        self._odeCompile = None
        # and we assume initially that we don't want the Jacobian
        self._Jacobian = None
        self._JacobianCompile = None
        # wtf... why!
        self._diffJacobian = None
        self._diffJacobianCompile = None

        # Information... yea, what else would you expect
        self._Grad = None
        self._GradCompile = None
        # more information....
        self._GradJacobian = None
        self._GradJacobianCompile = None
        # More information!! ROAR! I think this is useless though
        # because this is the hessian of the ode which most of the
        # time we don't really care
        self._Hessian = None
        self._HessianWithParam = None

        # all the symbols that we need in order to compile
        # s = state + t
        # sp = state + t + param
        # the latter is required to compile the symbolic code
        # to the numeric setting
        self._s = self._stateList + [self._t]
        self._sp = self._s + self._paramList

        # information regarding the integration.  We want an internal
        # storage so we can invoke the plot method within the same class
        self._t0 = None
        self._x0 = None
        self._odeOutput = None
        self._odeSolution = None
        self._odeTime = None

        self._intName = None

        self._paramValue = [0]*len(self._paramList)
        # the class for shape re-adjustment. We would always like to
        # operate in the matrix form if possible as it takes up less
        # memory when operating, but the output is required to be of
        # the vector form
        self._SAUtil = ode_utils.shapeAdjust(self.num_state, self.num_param)
        # compile the code.  Note that we need the class because we
        # compile both the formatted and unformatted version.
        self._SC = ode_utils.compileCode()

    def __eq__(self, other):
        if isinstance(other, DeterministicOde):
            if self.get_ode_eqn() == other.get_ode_eqn():
                return True
            else:
                return False
        else:
            return False

    def __repr__(self):
        return "DeterministicOde" + self._get_model_str()

    ########################################################################
    #
    # Information about the ode
    #
    ########################################################################

    # TODO: check and see whether it is linear correctly!
    def linear_ode(self):
        '''
        To check whether the input ode is linear

        Returns
        -------
        bool
            True if it is linear, False otherwise
        '''
        # we always assume that it is true to begin with
        # if the ode is linear, then a numerical integration
        # scheme is a waste of time
        is_linear = True
        # if we do not current possess the jacobian, we find it! ROAR!
        if self._Jacobian is None:
            self.get_jacobian_eqn()

        # a really stupid way to determining whether it is linear.
        # have not figured out a better way yet...
        a = self._Jacobian.atoms()
        for s in self._stateDict.values():
            if s in a:
                is_linear = False
#         for i in range(0, self._numState):
#             for j in range(0, self._numState):
#                 for k in range(0, self._numState):
#                     if self._Jacobian[i,j].has(self._stateList[k]):
#                         isLinear = False

        return is_linear

    # TODO: To check whether we have a DAE or just an ODE
    # def isDAE(self):
    #     return None

    # TODO: To find out whether there are situation where the
    # jacobian is actually singular, i.e. if it can be a DAE
    # def canDAE(self,x0,t0):

    ########################################################################
    #
    # Information about the ode
    #
    ########################################################################

    def get_ode_eqn(self, param_sub=False):
        '''
        Find the algebraic equations of the ode system.

        Returns
        -------
        :class:`sympy.matrices.matrices`
            ode in matrix form

        '''

        if self._ode is None:
            self._findOde()
        elif self._hasNewTransition.ode:
            self._findOde()
        else:
            pass

        if param_sub:
            return self._ode.subs(self._parameters)
        else:
            return self._ode

    def print_ode(self, latex_output=False):
        '''
        Prints the ode in symbolic form onto the screen/console in actual
        symbols rather than the word of the symbol.

        Parameters
        ----------
        latex_output: bool, optional
            Defaults to false which prints the equation in terms of symbols,
            if set to yes then the formula in terms of latex equations will
            be printed onto the screen.
        '''
        A = self.get_ode_eqn()
        B = sympy.zeros(A.rows,2)
        for i in range(A.shape[0]):
            B[i,0] = sympy.symbols('d' + str(self._stateList[i]) + '/dt=')
            B[i,1] = A[i]

        if latex_output:
            print(sympy.latex(B, mat_str="array", mat_delim=None,
                              inv_trig_style='full'))
        else:
            sympy.pretty_print(B)

    def _findOde(self):
        # lets see how we have defined our ode
        # if it is explicit, then we go straight to the easy case
        if self._explicitOde:
            # we have explicit ode and we should obtain them directly
            super(DeterministicOde, self)._computeOdeVector()
        else:
            # super(DeterministicOde, self)._computeTransitionMatrix()
            # super(DeterministicOde, self)._computeTransitionVector()
            # convert the transition matrix into the set of ode
            self._ode = sympy.zeros(self.num_state, 1)
            pureTransitionList = self._getAllTransition(pureTransitions=True)
            fromList, \
                to, \
                eqn = self._unrollTransitionList(pureTransitionList)
            for i, eqn in enumerate(eqn):
                for k in fromList[i]:
                    self._ode[k] -= eqn
                for k in to[i]:
                    self._ode[k] += eqn

        # now we just need to add in the birth death processes
        super(DeterministicOde, self)._computeBirthDeathVector()
        self._ode += self._birthDeathVector

        self._s = [s for s in self._iterStateList()] + [self._t]
        self._sp = self._s + [p for p in self._iterParamList()]

        # tests to see whether we have an autonomous system.  Need to
        # convert a non-autonmous system into an autonomous.  Note that
        # we will not do the conversion internally and require the
        # user to do this.  May consider this a feature in the future.
        for i, eqn in enumerate(self._ode):
            if self._t in eqn.atoms():
                raise Exception("Input is a non-autonomous system. " +
                                "We can only deal with an autonomous " +
                                "system at this moment in time")

            self. _ode[i], isDifficult = simplifyEquation(eqn)
            self._isDifficult = self._isDifficult or isDifficult

        if self._isDifficult:
            self._odeCompile = self._SC.compileExprAndFormat(self._sp,
                                                             self._ode,
                                                             modules='mpmath',
                                                             outType="vec")
        else:
            self._odeCompile = self._SC.compileExprAndFormat(self._sp,
                                                             self._ode,
                                                             outType="vec")
        # assign None to all others because we have reset the set of equations.
        self._hasNewTransition.trip()
        self._Grad = None
        self._Hessian = None
        self._Jacobian = None
        self._diffJacobian = None

        # happy!
        self._hasNewTransition.reset('ode')

        return self._ode

    def get_transition_graph(self, file_name=None, show=True):
        '''
        Returns the transition graph using graphviz

        Parameters
        ----------
        file_name: str, optional
            name of the output file, defaults to None
        show: bool, optional
            If the graph should be plotted, defaults to True

        Returns
        -------
        :class:`graphviz.Digraph`
        '''
        dot = _ode_composition.generateTransitionGraph(self, file_name)
        if show:
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
            img = mpimg.imread(io.BytesIO(dot.pipe("png")))
            plt.imshow(img)
            plt.show(block=False)
            return dot
        else:
            return dot

    #
    # this is the main ode solver
    #
    def ode(self, state, t):
        '''
        Evaluate the ode given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            output of the same length as the ode

        '''
        return self.eval_ode(time=t, state=state)

    def ode_T(self, t, state):
        '''
        Same as :meth:`ode` but with t as the first parameter
        '''
        return self.ode(state, t)

    def eval_ode(self, parameters=None, time=None, state=None):
        """
        Evaluate the ode given time, state and parameters.  An extension
        of :meth:`ode` but now also include the parameters.

        Parameters
        ----------
        parameters: list
            see :meth:`.parameters`
        time: numeric
            The current time
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            output of the same length as the ode.

        Notes
        -----
        There are differences between the output of this function and
        :meth:`.ode`.  Name and order of state and time are also
        different.

        See Also
        --------
        :meth:`.ode`

        """
        if self._ode is None or self._hasNewTransition.ode:
            self.get_ode_eqn()

        eval_param = self._getEvalParam(state, time, parameters)
        return self._odeCompile(eval_param)


    ########################################################################
    #
    # jacobian related operations
    #
    ########################################################################

    def is_stiff(self, state=None, t=None):
        '''
        Test on the eigenvalues of the jacobian.  We classify the
        problem as stiff if any of the eigenvalues are positive

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            eigenvalues of the system given input

        '''
        e = self.jacobian_eigenvalue(state, t)
        return np.any(e > 0)

    def jacobian_eigenvalue(self, state=None, t=None):
        '''
        Find out the eigenvalues of the jacobian given state and time. If
        None is given, the initial values are used.

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        bool
            True if any eigenvalue is positive

        '''

        if state is None or t is None:
            if self._x0 is not None and self._t0 is not None:
                J = self.jacobian(self._x0, self._t0)
        else:
            J = self.jacobian(state, t)

        return scipy.linalg.eig(J)[0]

    def jacobian(self, state, t):
        '''
        Evaluate the jacobian given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix of dimension [number of state x number of state]

        '''
        return self.eval_jacobian(time=t, state=state)

    def jacobian_T(self, t, state):
        '''
        Same as :meth:`jacobian` but with t as first parameter
        '''
        return self.jacobian(state, t)

    def _Jacobian_NoCheck(self, state, t):
        return self._evalJacobian_NoCheck(time=t, state=state)

    def _JacobianT_NoCheck(self, t, state):
        return self._Jacobian_NoCheck(state, t)

    def get_jacobian_eqn(self):
        '''
        Returns the jacobian in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of state]

        '''
        if self._Jacobian is None:
            self.get_ode_eqn()
            states = [s for s in self._iterStateList()]
            self._Jacobian = self._ode.jacobian(states)
            for i in range(self.num_state):
                for j in range(self.num_state):
                    eqn = self._Jacobian[i,j]
                    if  eqn != 0:
                        self._Jacobian[i,j], isDifficult = simplifyEquation(eqn)
                        self._isDifficult = self._isDifficult or isDifficult

        f = self._SC.compileExprAndFormat
        if self._isDifficult:
            self._JacobianCompile = f(self._sp,
                                      self._Jacobian,
                                      modules='mpmath')
        else:
            self._JacobianCompile = f(self._sp,
                                      self._Jacobian)

        self._hasNewTransition.reset('Jacobian')

        return self._Jacobian

    def eval_jacobian(self, parameters=None, time=None, state=None):
        '''
        Evaluate the jacobian given parameters, state and time. An extension
        of :meth:`.jacobian` but now also include the parameters.

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
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of state x number of state]

        Notes
        -----
        Name and order of state and time are also different.

        See Also
        --------
        :meth:`.jacobian`

        '''
        if self._Jacobian is None or self._hasNewTransition.Jacobian:
            #self.get_ode_eqn()
            self.get_jacobian_eqn()

        eval_param = self._getEvalParam(state, time, parameters)
        return self._JacobianCompile(eval_param)

    def _evalJacobian_NoCheck(self, time, state):
        '''
        Same as :meth:`eval_jacobian` but without the checks
        '''
        eval_param = list(state) + [time] + self._paramValue
        return self._JacobianCompile(eval_param)

    ######  the sum of jacobian, i.e a_{i} = \sum_{j=1}^{d} J_{i,j}

    def sens_jacobian_state(self, state_param, t):
        '''
        Evaluate the jacobian of the sensitivity w.r.t. the
        state given state and time

        Parameters
        ----------
        state_param: array like
            The current numerical value for the states as
            well as the sensitivities, which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix of dimension [number of state *
            number of parameters x number of state]

        '''

        state = state_param[0:self.num_state]
        sens = state_param[self.num_state::]

        return self.eval_sens_jacobian_state(time=t, state=state, sens=sens)

    def sens_jacobian_state_T(self, t, state):
        '''
        Same as :meth:`sens_jacobian_state_T` but with t as first parameter
        '''
        return self.sens_jacobian_state(state, t)

    def eval_sens_jacobian_state(self, time=None, state=None, sens=None):
        '''
        Evaluate the jacobian of the sensitivities w.r.t the states given
        parameters, state and time. An extension of :meth:`.sens_jacobian_state`
        but now also include the parameters.

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
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of state x number of state]

        Notes
        -----
        Name and order of state and time are also different.

        See Also
        --------
        :meth:`.sens_jacobian_state`

        '''

        nS = self.num_state
        nP = self.num_param

        # dot first, then transpose, then reshape
        # basically, some magic
        # don't ask me what is actually going on here, I did it
        # while having my wizard hat on
        return(np.reshape(self.diff_jacobian(state, time).dot(
            self._SAUtil.vecToMatSens(sens)).transpose(), (nS*nP, nS)))

    ############################## derivative of jacobian

    def diff_jacobian(self, state, t):
        '''
        Evaluate the differential of jacobian given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix of dimension [number of state x number of state]

        '''
        return self.eval_diff_jacobian(time=t, state=state)

    def diff_jacobian_T(self, t, state):
        '''
        Same as :meth:`diff_jacobian` but with t as first parameter
        '''
        return self.diff_jacobian(state, t)

    def get_diff_jacobian_eqn(self):
        '''
        Returns the jacobian differentiate w.r.t. states in algebraic form

        Returns
        -------
        list
            list of size (num of state,) each with
            :mod:`sympy.matrices.matrices` of dimension
            [number of state x number of state]

        '''
        if self._diffJacobian is None:
            self.get_ode_eqn()
            diffJac = list()

            for eqn in self._ode:
                J = sympy.zeros(self.num_state, self.num_state)
                for i, si in enumerate(self._iterStateList()):
                    diffEqn, D1 = simplifyEquation(diff(eqn, si, 1))
                    for j, sj in enumerate(self._iterStateList()):
                        J[i,j], D2 = simplifyEquation(diff(diffEqn, sj, 1))
                        self._isDifficult = self._isDifficult or D1 or D2
                #binding.
                diffJac.append(J)

            # extract first matrix as base.  we have to get the first element
            # as base if we want to use the class method of the object
            diffJacMatrix = diffJac[0]
            for i in range(1, len(diffJac)):
                # sympy internal matrix joining
                diffJacMatrix = diffJacMatrix.col_join(diffJac[i])

            self._diffJacobian = copy.deepcopy(diffJacMatrix)

        f = self._SC.compileExprAndFormat
        if self._isDifficult:
            self._diffJacobianCompile = f(self._sp,
                                          self._diffJacobian,
                                          modules='mpmath')
        else:
            self._diffJacobianCompile = f(self._sp,
                                          self._diffJacobian)

        self._hasNewTransition.reset('diffJacobian')

        return self._diffJacobian

    def eval_diff_jacobian(self, parameters=None, time=None, state=None):
        '''
        Evaluate the differential of the jacobian given parameters,
        state and time. An extension of :meth:`.diff_jacobian` but now
        also include the parameters.

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
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of state x number of state]

        Notes
        -----
        Name and order of state and time are also different.

        See Also
        --------
        :meth:`.jacobian`

        '''
        if self._diffJacobian is None or self._hasNewTransition.diffJacobian:
            #self.get_ode_eqn()
            self.get_diff_jacobian_eqn()

        eval_param = self._getEvalParam(state, time, parameters)
        return self._diffJacobianCompile(eval_param)

    ########################################################################
    #
    # Gradient related operations
    #
    ########################################################################

    def get_grad_eqn(self):
        '''
        Return the gradient of the ode in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of parameters]

        '''
        # finds

        if self._Grad is None:
            ode = self.get_ode_eqn()
            self._Grad = sympy.zeros(self.num_state, self.num_param)

            for i in range(self.num_state):
                # need to adjust such that the first index is not
                # included because it correspond to time
                for j, p in enumerate(self._iterParamList()):
                    eqn, isDifficult = simplifyEquation(diff(ode[i], p, 1))
                    self._Grad[i,j] = eqn
                    self._isDifficult = self._isDifficult or isDifficult

        if self._isDifficult:
            self._GradCompile = self._SC.compileExprAndFormat(self._sp,
                                                              self._Grad,
                                                              modules='mpmath',
                                                              outType="mat")
        else:
            self._GradCompile = self._SC.compileExprAndFormat(self._sp,
                                                              self._Grad,
                                                              outType="mat")
        self._hasNewTransition.reset('grad')

        return self._Grad

    def grad(self, state, time):
        """
        Evaluate the gradient given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: numeric
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix of dimension [number of state x number of parameters]

        """
        return self.eval_grad(state=state, time=time)

    def grad_T(self, t, state):
        '''
        Same as :meth:`grad_T` but with t as first parameter
        '''
        return self.grad(state, t)

    def eval_grad(self, parameters=None, time=None, state=None):
        '''
        Evaluate the gradient given parameters, state and time. An extension
        of :meth:`grad` but now also include the parameters.

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
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of state x number of state]

        Notes
        -----
        Name and order of state and time are also different.

        See Also
        --------
        :meth:`.grad`

        '''
        if self._Grad is None or self._hasNewTransition.grad:
            #self.get_ode_eqn()
            self.get_grad_eqn()

        eval_param = self._getEvalParam(state, time, parameters)
        return self._GradCompile(eval_param)

    #
    # jacobian of the Gradiant
    #

    def get_grad_jacobian_eqn(self):
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
        if self._GradJacobian is None:
            self._GradJacobian = sympy.zeros(self.num_state*self.num_param,
                                             self.num_state)
            G = self.get_grad_eqn()
            for k in range(0, self.num_param):
                for i in range(0, self.num_state):
                    for j, s in enumerate(self._iterStateList()):
                        z = k*self.num_state + i
                        eqn, isDifficult = simplifyEquation(diff(G[i,k], s, 1))
                        self._GradJacobian[z,j] = eqn
                        self._isDifficult = self._isDifficult or isDifficult
            # end of the triple loop.  All elements are now filled

        f = self._SC.compileExprAndFormat
        if self._isDifficult:
            self._GradJacobianCompile = f(self._sp,
                                          self._GradJacobian,
                                          modules='mpmath')
        else:
            self._GradJacobianCompile = f(self._sp,
                                         self._GradJacobian)

        self._hasNewTransition.reset('GradJacobian')

        return self._GradJacobian

    def grad_jacobian(self, state, time):
        """
        Evaluate the Jacobian of the gradient given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: numeric
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix of dimension [number of state x number of parameters]

        See also
        --------
        :meth:`.grad`

        """
        return self.eval_grad_jacobian(state=state, time=time)

    def grad_jacobianT(self, t, state):
        '''
        Same as :meth:`grad_jacobian` but with t as first parameter
        '''
        return self.grad_jacobian(state, t)

    def eval_grad_jacobian(self, parameters=None, time=None, state=None):
        '''
        Evaluate the jacobian of the gradient given parameters,
        state and time. An extension of :meth:`.grad_jacobian`
        but now also include the parameters.

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
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of state x number of state]

        Notes
        -----
        Name and order of state and time are also different.

        See Also
        --------
        :meth:`.grad_jacobian`, :meth:`.get_grad_jacobian_eqn`

        '''
        if self._GradJacobian is None or self._hasNewTransition.GradJacobian:
            #self.get_ode_eqn()
            self.get_grad_jacobian_eqn()

        eval_param = self._getEvalParam(state, time, parameters)
        return self._GradJacobianCompile(eval_param)

    ########################################################################
    #
    # hessian related operations
    #
    ########################################################################

    def get_hessian_eqn(self):
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

        if self._Hessian is None:
            ode = self.get_ode_eqn()
            self._Hessian = list()
            # roll out the equation one by one.  Each H below is a the
            # second derivative of f_{j}(x), the j^{th} ode.  Each ode
            # correspond to a state
            for eqn in ode:
                H = sympy.zeros(self.num_param, self.num_param)
                # although this can be simplified by first finding the gradient
                # it is not required so we will be slow here
                for i, pi in enumerate(self._iterParamList()):
                    a = diff(eqn, pi, 1)
                    for j, pj in enumerate(self._iterParamList()):
                        H[i,j], isDifficult = simplifyEquation(diff(a, pj, 1))
                        self._isDifficult = self._isDifficult or isDifficult
                # end of double loop.  Finished one state
                self._Hessian.append(H)

        return self._Hessian

    def hessian(self, state, time):
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
        of :meth:`hessian` but now also include the parameters.

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
            self.get_ode_eqn()

        eval_param = list()
        eval_param = self._addTimeEvalParam(eval_param, time)
        eval_param = self._addStateEvalParam(eval_param, state)

        if parameters is None:
            if self._HessianWithParam is None:
                self._computeHessianParam()
        else:
            self.parameters = parameters

        if self._Hessian is None:
            self._computeHessianParam()

        if len(eval_param) == 0:
            return self._Hessian
        else:
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

    ########################################################################
    #
    # Sensitivity related operations (1st forward)
    #
    ########################################################################

    def sensitivity(self, sens, t, state, by_state=False):
        """
        Evaluate the sensitivity given state and time.  The default is to
        output the values by parameters, i.e. :math:`s_{i},\\ldots,s_{i+n}` are
        partial derivatives w.r.t. the states for
        :math:`i \\in {1,1+p,1+2p,1+3p, \\ldots, 1+(n-1)p}`.  This is
        to take advantage of the fact that we have a block diagonal
        jacobian that was already evaluated

        Parameters
        ----------
        sens: array like
            The starting sensitivity of size [number of state x number of
            parameters].  Which are normally zero or one,
            depending on whether the initial conditions are also variables.
        t: double
            The current time
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        by_state: bool
            how we want the output to be arranged.  Default is True so
            that we have a block diagonal structure

        Returns
        -------
        :class:`numpy.ndarray`
        """
        # TODO: allows the functionality to not evaluate all sensitivity

        # S = \nabla_{time} \frac{\partial State}{\partial Parameters}
        # rearrange the input if required
        if by_state:
            S = np.reshape(sens, (self.num_state, self.num_param))
        else:
            S = self._SAUtil.vecToMatSens(sens)

        return self.eval_sensitivity(S=S, t=t, state=state, by_state=by_state)

    def sensitivity_T(self, t, sens, state, by_state=False):
        '''
        Same as :meth:`sensitivity` but with t as first parameter
        '''
        return self.sensitivity(sens, t, state, by_state)

    def eval_sensitivity(self, S, t, state, by_state=False):
        """
        Evaluate the sensitivity given state and time

        Parameters
        ----------
        S: array like
            Which should be :class:`numpy.ndarray`.
            The starting sensitivity of size [number of state x number of
            parameters].  Which are normally zero or one,
            depending on whether the initial conditions are also variables.
        t: double
            The current time
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        by_state: bool
            how we want the output to be arranged.  Default is True so
            that we have a block diagonal structure

        Returns
        -------
        :class:`numpy.ndarray`

        Notes
        -----
        It is different to :meth:`.eval_ode` and :meth:`.eval_jacobian` in
        that the extra input argument is not a parameter

        See Also
        --------
        :meth:`.sensitivity`

        """

        # jacobian * sensitivities + G
        # where G is the gradient
        J = self.jacobian(state, t)
        G = self.grad(state, t)
        A = np.dot(J, S) + G

        if by_state:
            return np.reshape(A, self.num_state*self.num_param)
        else:
            return self._SAUtil.matToVecSens(A)

    def ode_and_sensitivity(self, state_param, t, by_state=False):
        '''
        Evaluate the sensitivity given state and time

        Parameters
        ----------
        state_param: array like
            The current numerical value for the states as well as the
            sensitivities values all in one.  We assume that the state
            values comes first.
        t: double
            The current time
        by_state: bool
            Whether the output vector should be arranged by state or by
            parameters. If False, then it means that the vector of output is
            arranged according to looping i,j from Sensitivity_{i,j} with i
            being the state and j the param. This is the preferred way because
            it leds to a block diagonal Jacobian

        Returns
        -------
        :class:`list`
            concatenation of 2 element. First contains the ode, second the
            sensitivity. Both are of type :class:`numpy.ndarray`

        See Also
        --------
        :meth:`.sensitivity`, :meth:`.ode`

        '''

        if len(state_param) == self.num_state:
            raise InputError("You have only inputed the initial condition " +
                             "for the states and not the sensitivity")

        # unrolling, assuming that we would always put the state first
        # there is no safety checks on this because it is impossible to
        # distinguish what is state and what is sensitivity as they are
        # all numeric value that can take the full range (-\infty,\infty)
        state = state_param[0:self.num_state]
        sens = state_param[self.num_state::]

        out1 = self.ode(state, t)
        out2 = self.sensitivity(sens, t, state, by_state)
        return np.append(out1, out2)

    def ode_and_sensitivity_T(self, t, state_param, by_state=False):
        '''
        Same as :meth:`ode_and_sensitivity` but with t as first parameter
        '''
        return self.ode_and_sensitivity(state_param, t, by_state)

    def ode_and_sensitivity_jacobian(self, state_param, t, by_state=False):
        '''
        Evaluate the sensitivity given state and time.  Output a block
        diagonal sparse matrix as default.

        Parameters
        ----------
        state_param: array like
            The current numerical value for the states as well as the
            sensitivities values all in one.  We assume that the state
            values comes first.
        t: double
            The current time
        by_state: bool
            How the output is arranged, according to the vector of output.
            It can be in terms of state or parameters, where by state means
            that the jacobian is a block diagonal matrix.

        Returns
        -------
        :class:`numpy.ndarray`
            output of a square matrix of size: number of ode + 1 times number
            of parameters

        See Also
        --------
        :meth:`.ode_and_sensitivity`

        '''

        if len(state_param) == self.num_state:
            raise InputError("Expecting both the state and the sensitivities")
        else:
            state = state_param[0:self.num_state]

        # now we start the computation
        J = self.jacobian(state, t)
        # create the block diagonal Jacobian, assuming that whoever is
        # calling this function wants it arranges by state-parameters

        # Note that none of the ode integrator in scipy allow a sparse Jacobian
        # matrix.  All of them accept a banded matrix in packed format but not
        # an actual sparse, or specifying the number of bands.
        outJ = np.kron(np.eye(self.num_param), J)
        # Jacobian of the gradient
        GJ = self.grad_jacobian(state, t)
        # and now we add the gradient
        sensJacobianOfState = GJ + self.sens_jacobian_state(state_param, t)

        if by_state:
            arrangeVector = np.zeros(self.num_state * self.num_param)
            k = 0
            for j in range(0, self.num_param):
                for i in range(0, self.num_state):
                    if i == 0:
                        arrangeVector[k] = (i*self.num_state) + j
                    else:
                        arrangeVector[k] = (i*(self.num_state - 1)) + j
                    k += 1

            outJ = outJ[np.array(arrangeVector,int),:]
            idx = np.array(arrangeVector, int)
            sensJacobianOfState = sensJacobianOfState[idx,:]
        # The Jacobian of the ode, then the sensitivities w.r.t state and
        # the sensitivities. In block form.  Theoretically, only the diagonal
        # blocks are important but we output the full matrix for completeness
        return np.asarray(np.bmat([
            [J, np.zeros((self.num_state, self.num_state*self.num_param))],
            [sensJacobianOfState, outJ]
        ]))

    def ode_and_sensitivity_jacobian_T(self, t, state_param, by_state=False):
        '''
        Same as :meth:`ode_and_sensitivity_jacobian` but with t as
        first parameter
        '''
        return self.ode_and_sensitivity_jacobian(state_param, t, by_state)

    ########################################################################
    #
    # Include initial value as parameters. Sensitivity related operations
    # (1st forward)
    #
    ########################################################################

    def sensitivityIV(self, sensIV, t, state):
        """
        Evaluate the sensitivity which include the initial values as
        our parameters given state and time.  The default is to
        output the values by parameters, i.e. :math:`s_{i},\\ldots,s_{i+n}` are
        partial derivatives w.r.t. the states for
        :math:`i \\in {1,1+p,1+2p,1+3p, \\ldots, 1+(n-1)p}`.  This is to take
        advantage of the fact that we have a block diagonal Jacobian that was
        already evaluated.

        Parameters
        ----------
        sensIV: array like
            The starting sensitivity of size [number of state x number of
            parameters] + [number of state x number of state] for the
            initial condition.  The latter is an identity matrix at time zero.
        t: double
            The current time
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.ndarray`
            output of the same length as the ode

        """

        nS = self.num_state
        nP = self.num_param
        # separate information out.  Again, we do have not have checks here
        # as it will be impossible to distinguish what is correct
        sens = sensIV[:(nS*nP)]
        S = self._SAUtil.vecToMatSens(sens)

        IV = np.reshape(sensIV[-(nS*nS):], (nS, nS), 'F')

        return self.eval_sensitivityIV(S=S, IV=IV, t=t, state=state)

    def sensitivityIV_T(self, t, sensIV, state):
        '''
        Same as :meth:`sensitivityIV` but with t as first parameter
        '''
        return self.sensitivityIV(sensIV, t, state)

    def eval_sensitivityIV(self, S, IV, t, state):
        """
        Evaluate the sensitivity with initial values given
        state and time

        Parameters
        ----------
        S: array like
            Which should be :class:`numpy.ndarray`.
            The starting sensitivity of size [number of state x number of
            parameters].  Which are normally zero or one,
            depending on whether the initial conditions are also variables.
        IV: array like
            sensitivities for the initial values
        t: double
            The current time
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.ndarray`
            :math:`f(s(x,\\theta))` and :math:`f(s(x_{0}))`

        Notes
        -----
        It is different to :meth:`.eval_ode` and :meth:`.eval_jacobian` in
        that the extra input argument is not a parameter.

        See Also
        --------
        :meth:`.sensitivityIV`

        """
        # jacobian * sensitivities + G
        # where G is the gradient
        # Evidently, A below uses the same operations as
        # A = self.eval_sensitivity(S,t,state)
        # but we are evaluating them explicitly here because
        # we will be using J as well when computing B

        J = self.jacobian(state, t)
        G = self.grad(state, t)
        A = np.dot(J, S) + G

        # and jacobian * sensitivities of the initial condition
        B = np.dot(J, IV)

        # we want to output by parameters
        return self._SAUtil.matToVecSens(A), B.flatten('F')

    def ode_and_sensitivityIV(self, state_param, t):
        '''
        Evaluate the sensitivity given state and time

        Parameters
        ----------
        state_param: array like
            The current numerical value for the states as well as the
            sensitivities values all in one.  We assume that the state
            values comes first.
        t: double
            The current time

        Returns
        -------
        :class:`list`
            concatenation of 3 element. First contains the ode, second the
            sensitivity, then the sensitivity of the initial value.  All
            of them are of type
            :class:`numpy.ndarray`

        See Also
        --------
        :meth:`.sensitivity`, :meth:`.ode`

        '''

        if len(state_param) == self.num_state:
            raise InputError("You have only inputed the initial condition " +
                             "for the states and not the sensitivity")

        # unrolling, assuming that we would always put the state first
        state = state_param[0:self.num_state]
        # the remainings
        sens_iv = state_param[self.num_state::]
        # separate evaluation
        out1 = self.ode(state,t)
        out2,out3 = self.sensitivityIV(sens_iv, t, state)
        return np.append(np.append(out1, out2), out3)

    def ode_and_sensitivityIV_T(self, t, state_param):
        '''
        Same as :meth:`ode_and_sensitivityIV` but with t as first parameter
        '''
        return self.ode_and_sensitivityIV(state_param, t)

    def ode_and_sensitivityIV_jacobian(self, state_param, t):
        '''
        Evaluate the sensitivity given state and time.  Output a block
        diagonal sparse matrix as default.

        Parameters
        ----------
        state_param: array like
            The current numerical value for the states as well as the
            sensitivities values all in one.  We assume that the state
            values comes first.
        t: double
            The current time
        byState: bool
            How the output is arranged, according to the vector of output.
            It can be in terms of state or parameters, where by state means
            that the jacobian is a block diagonal matrix.

        Returns
        -------
        :class:`numpy.ndarray`
            output of a square matrix of size: number of ode + 1 times number
            of parameters

        See Also
        --------
        :meth:`.ode_and_sensitivity`

        '''

        if len(state_param) == self.num_state:
            raise InputError("Expecting both the state and the sensitivities")
        else:
            state = state_param[0:self.num_state]

        nS = self.num_state
        nP = self.num_param
        # now we start the computation, the simply one :)
        J = self.jacobian(state, t)

        # now the jacobian of the state vs initial value
        DJ = self.diff_jacobian(state, t)
        A = DJ.dot(np.reshape(state_param[(nS*(nP+1))::], (nS, nS), 'F'))
        A = np.reshape(A.transpose(), (nS*nS, nS))

        if nP == 0:
            return np.asarray(np.bmat([
                        [J, np.zeros((nS, nS*nS))],
                        [A, np.kron(np.eye(nS), J)]
                        ]))
        else:
            # create the block diagonal jacobian, assuming that whoever is
            # calling this function wants it arranges by state-parameters
            outJ = np.kron(np.eye(nP), J)

            # jacobian of the gradient
            GJ = self.grad_jacobian(state, t)
            GS = self.sens_jacobian_state(state_param[:(nS*(nP + 1))], t)
            sensJacobianOfState = GJ + GS

            # The jacobian of the ode, then the sensitivities w.r.t state
            # and the sensitivities. In block form
            return np.asarray(np.bmat([
                [J, np.zeros((nS, nS*nP)), np.zeros((nS, nS*nS))],
                [sensJacobianOfState, outJ, np.zeros((nS*nP, nS*nS))],
                [A, np.zeros((nS*nS, nS*nP)), np.kron(np.eye(nS), J)]
            ]))

    def ode_and_sensitivityIV_jacobian_T(self, t, state_param):
        '''
        Same as :meth:`ode_and_sensitivityIV_jacobian` but with t as
        first parameter
        '''
        return self.ode_and_sensitivityIV_jacobian(state_param, t)

    ############################################################################
    #
    # Adjoint
    #
    ############################################################################

    def adjoint_interpolate(self, state, t, interpolant, func=None):
        '''
        Compute the adjoint given the adjoint vector, time, the functions
        which was used to interpolate the state variable

        Parameters
        ----------
        state: array like
            The current value of lambda, where lambda's are the Lagrangian
            multipliers of the differential equation.
        t: double
            The current time.
        interpolant: list
            list of interpolating functions of the state
        func: callable
            This should take inputs similar to an ode, i.e. of the form
            func(y,t).  If j(y,t) is the cost function, then func
            is a function that calculates
            :math:`\\partial j \\over \\partial x`.

        Returns
        -------
        :class:`numpy.ndarray`
            output of the same length as the ode
        '''
        state_param = [o(t) for o in interpolant]
        return self.adjoint(state, t, state_param, func)

    def adjoint_interpolate_T(self, t, state, interpolant, objInput=None):
        '''
        Same as :meth:`adjoint_interpolate` but with t as first parameter
        '''
        return self.adjoint_interpolate(state, t, interpolant, objInput)

    def _adjointInterpolate_NoCheck(self, state, t,
                                    interpolant, func=None):
        state_param = [o(t) for o in interpolant]
        return self._adjoint_NoCheck(state, t, state_param, func)

    def _adjointInterpolateT_NoCheck(self, t, state,
                                     interpolant, func=None):
        return self._adjoint_NoCheck(state, t, interpolant, func)

    def adjoint(self, state, t, state_param, func=None):
        '''
        Compute the adjoint given the adjoint vector, time, state variable
        and the objective function.  Note that this function is very
        restrictive in the sense that the (original) state variable changes
        through time but this assumes it is a constant, i.e. we assume that
        the original system is linear.

        Parameters
        ----------
        state: array like
            The current value of lambda, where lambda's are the Lagrangian
            multipliers of the differential equation.
        t: double
            The current time.
        state_param: array like
            The state vector that is (or maybe) required to evaluate the
            jacobian of the original system
        func: callable
            This should take inputs similar to an ode, i.e. of the form
            func(y,t).  If j(y,t) is the cost function, then func
            is a function that calculates
            :math:`\\partial j \\over \\partial x`.

        Returns
        -------
        :class:`numpy.ndarray`
            output of the same length as the ode

        Notes
        -----
        The size of lambda should be the same as the state. The integral
        should be starting from T, the final time of the original system
        and is integrated backwards (for stability).

        '''
        J = self.jacobian(state_param, t)

        if func is None:
            return np.dot(state, -J)
        else:
            return func(state_param, t) - J.transpose().dot(state)

    def _adjoint_NoCheck(self, state, t, state_param, func=None):
        J = self._Jacobian_NoCheck(state_param, t)
        if func is None:
            return np.dot(state, -J)
        else:
            return func(state_param, t) - J.transpose().dot(state)

    def _adjoinT_NoCheck(self, t, state, state_param, func=None):
        return self._adjoint_NoCheck(state, t, state_param, func)

    def adjoint_T(self, t, state, state_param, func=None):
        '''
        Same as :meth:`adjoint` but with t as first parameter
        '''
        return self.adjoint(state, t, state_param, func)

    def adjoint_jacobian(self, state, t, state_param, func=None):
        '''
        Compute the jacobian of the adjoint given the adjoint vector, time,
        state variable and the objective function.  This is simply the same
        as the negative jacobian of the ode transposed.

        Parameters
        ----------
        state: array like
            The current value of lambda, where lambda's are the Lagrangian
            multipliers of the differential equation.
        t: double
            The current time.
        state_param: array like
            The state vector that is (or maybe) required to evaluate the
            jacobian of the original system
        func: callable
            This should take inputs similar to an ode, i.e. of the form
            func(y,t).  If j(y,t) is the cost function, then func
            is a function that calculates
            :math:`\\partial j \\over \\partial x`.

        Returns
        -------
        :class:`numpy.ndarray`
            output of is a two dimensional array of size
            [number of state x number of state]

        Notes
        -----
        It takes the same number of argument as the adjoint for simplicity
        when integrating.

        See Also
        --------
        :meth:`.adjoint`

        '''
        return -self.jacobian(state_param, t).transpose()

    def adjoint_jacobian_T(self, t, state, state_param, func=None):
        '''
        Same as :meth:`adjoint_jacobian_T` but with t being the
        first parameter
        '''
        return self.adjoint_jacobian(state, t, state_param, func)

    def adjoint_interpolate_jacobian(self, state, t,
                                     interpolant, func=None):
        '''
        Compute the Jacobian of the adjoint given the adjoint vector, time,
        function of the interpolation on the state variables and the
        objective function.  This is simply the same as the negative
        Jacobian of the ode transposed.

        Parameters
        ----------
        state: array like
            The current value of lambda, where lambda's are the Lagrangian
            multipliers of the differential equation.
        t: double
            The current time.
        interpolant: list
            list of interpolating functions of the state
        func: callable
            This should take inputs similar to an ode, i.e. of the form
            func(y,t).  If j(y,t) is the cost function, then func is
            a function that calculates :math:`\\partial j \\over \\partial x`.

        Returns
        -------
        :class:`numpy.ndarray`
            output of is a two dimensional array of size
            [number of state x number of state]

        Notes
        -----
        Same as :meth:`.adjoint_jacobian` but takes a list of interpolating
        function instead of a single (vector) value

        See Also
        --------
        :meth:`.adjoint_jacobian`

        '''
        state_param = [o(t) for o in interpolant]
        return self.adjoint_jacobian(state, t, state_param, func)

    def adjoint_interpolate_jacobian_T(self, t, state, interpolant, func=None):
        '''
        Same as :meth:`adjoint_interpolate_jacobian` but with t as
        first parameter
        '''
        return self.adjoint_interpolate_jacobian(state, t,
                                               interpolant, func)

    ########################################################################
    #
    # Sensitivity, forward-forward operations
    #
    ########################################################################

    def forwardforward(self, ff, t, state, s):
        '''
        Evaluate a single :math:`f(x)` of the forward-forward sensitivities

        Parameters
        ----------
        ff: array like
            the forward-forward sensitivities in vector form
        t: numeric
            time
        state: array like
            the current state
        s: array like
            forward sensitivities in vector form

        Returns
        -------
        :class:`numpy.ndarray`
            :math:`f(x)` of size [number of state *
            (number of parameters * number of parameters)]

        '''
        # byState is simply stupid in the forward forward case because the
        # second derivative makes things only rational if we look at it from
        # the state point of view
        S = self._SAUtil.vecToMatSens(s)
        FF = self._SAUtil.vecToMatFF(ff)
        return self.eval_forwardforward(FF=FF, S=S, state=state, t=t)

    def forwardforward_T(self, t, ff, s, state):
        '''
        Same as :meth:`forwardforward` but with t as the first
        parameter
        '''
        return self.forwardforward(ff, t, state, s)

    def eval_forwardforward(self, FF, S, state, t):
        '''
        Evaluate a single f(x) of the forward-forward sensitivities

        Parameters
        ----------
        FF: array like
            this is in fact a 3rd order Tensor, aka 3d array
        S: array like
            sensitivities in matrix form
        state: array like
            the current state
        t: numeric
            time

        Returns
        -------
        :class:`numpy.ndarray`
            f(x) of size [number of state *
            (number of parameters * number of parameters)]

        '''

        J = self.jacobian(state, t)
        diffJ = self.diff_jacobian(state, t)

        # evaluating by state/ode, the matrix of second derivative
        # we have kron products into all these evaluations and the class
        # here use a sparse matrix operation
        outFF = self._SAUtil.kronParam(J).dot(FF)
        outFF += self._SAUtil.kronState(A=S.T, pre=True).dot(diffJ).dot(S)

        # now we need to magic our list / matrix into a vector, aka append
        # each of the vectorized matrix one after another
        return self._SAUtil.matToVecFF(outFF)

    def ode_and_forwardforward(self, state_param, t):
        '''
        Evaluate a single f(x) of the ode and the
        forward-forward sensitivities

        Parameters
        ----------
        state_param: array like
            state and forward-forward sensitivities in vector form
        t: numeric
            time

        Returns
        -------
        :class:`numpy.ndarray`
            same size as the state_param input
        '''

        if len(state_param) == self.num_state:
            raise InputError("You have only inputed the initial condition " +
                             "for the states and not the sensitivity")
        elif len(state_param) == ((self.num_state + 1)*self.num_param):
            raise InputError("You have only inputed the initial condition " +
                             "for the states and the sensitivity but not " +
                             "the forward forward condition")

        # unrolling of parameters
        state = state_param[0:self.num_state]
        # we want the index up to numState * (numParam + 1)
        # as in, (numState * numParam + numState,
        # number of sensitivities + number of ode)
        sens = state_param[self.num_state:(self.num_state*(self.num_param + 1))]
        # the rest are then the forward forward sensitivities
        ff = state_param[(self.num_state*(self.num_param + 1))::]

        out1 = self.ode(state, t)
        out2 = self.sensitivity(sens, t, state)
        out3 = self.forwardforward(ff, t, state, sens)

        return np.append(np.append(out1, out2), out3)

    def ode_and_forwardforward_T(self, t, state_param):
        '''
        Same as :meth:`odeAndForwardForward` but with time
        as the first input

        '''
        return self.ode_and_forwardforward(state_param, t)

    def ode_and_forwardforward_jacobian(self, state_param, t):
        '''
        Return the jacobian after evaluation given the input
        of the state and the forward forward sensitivities

        Parameters
        ----------
        state_param: array like
            state and forward-forward sensitivities in vector form
        t: numeric
            time

        Returns
        -------
        :class:`numpy.ndarray`
            size of (a,a) where a is the length of the
            state_param input
        '''
        if len(state_param) == self.num_state:
            state = state_param
        else:
            state = state_param[0:self.num_state]

        J = self.jacobian(state, t)
        # create the block diagonal jacobian, assuming that whoever is
        # calling this function wants it arranges by state-parameters
        # We are only construct the block diagonal jacobian here
        # instead of the full one unlike some of the other methods within
        # this class
        outJS = np.kron(np.eye(self.num_param), J)
        outJFF = np.kron(np.eye(self.num_param*self.num_param), J)
        # The jacobian of the ode, then the sensitivities, then the
        # forward forward sensitivities
        return scipy.linalg.block_diag(J, outJS, outJFF)

    def ode_and_forwardforward_jacobian_T(self, t, state_param):
        '''
        Same as :meth:`ode_and_forwardforward_jacobian` but
        with t being the first parameters
        '''
        return self.ode_and_forwardforward_jacobian(state_param, t)

    ########################################################################
    #
    # Initial conditions, integrations and result plots
    #
    ########################################################################

    @property
    def initial_state(self):
        '''
        Return the initial state values
        '''
        return self._x0

    @initial_state.setter
    def initial_state(self, x0):
        '''
        Set the initial state values

        Parameters
        ----------
        x0: array like
             initial condition of x at time 0

        '''
        err_str = "More than one state in the defined system"

        if isinstance(x0, np.ndarray):
            self._x0 = x0
        elif isinstance(x0, (list, tuple)):
            self._x0 = np.array(x0)
        elif isinstance(x0, (int, float)):
            if self.num_state == 1:
                self._x0 = np.array([x0])
            else:
                raise InitializeError(err_str)
        else:
            raise InitializeError("err_str")

        if len(self._x0) != self.num_state:
            raise Exception("Number of state is " +
                            str(self.num_state)+ " but " +
                            str(len(self._x0))+ " detected")

    @property
    def initial_time(self):
        '''
        Return the initial time
        '''
        return self._t0

    @initial_time.setter
    def initial_time(self, t0):
        '''
        Set the initial time

        Parameters
        ----------
        t0: numeric
            initial time where x0 is observed

        '''

        err_str = "Initial time should be a "
        if isinstance(t0, Number):
            self._t0 = t0
        elif ode_utils.is_list_like(t0):
            if len(t0) == 1:
                if isinstance(t0[0], Number):
                    self._t0 = t0[0]
                else:
                    raise InitializeError(err_str + "numeric value")
            else:
                raise InitializeError(err_str + "single value")
        elif isinstance(t0, (list, tuple)):
            if len(t0) == 1:
                self._t0 = np.array(t0[0])
            else:
                raise InitializeError(err_str + "single value")
        else:
            raise InitializeError(err_str + "numeric value")

    @property
    def initial_values(self):
        '''
        Returns the initial values, both time and state as a tuple (x0, t0)
        '''
        return (self.initial_state, self.initial_time)

    @initial_values.setter
    def initial_values(self, x0t0):
        '''
        Set the initial values, both time and state

        Parameters
        ----------
        x0t0: array like
            initial condition of x at time t and the initial time t where x
            is observed
        '''
        assert len(x0t0) == 2, "Initial values require (x0, t0)"
        self.initial_state = x0t0[0]
        self.initial_time = x0t0[1]

    def integrate(self, t, full_output=False):
        '''
        Integrate over a range of t when t is an array and a output at time t

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
        full_output: bool
            if we want additional information
        '''
        # type checking
        self._setIntegrateTime(t)
        # if our parameters are stochastic, then we are going to generate
        # another set of parameters to run
        if self._stochasticParam is not None:
            # this should always be true.  If not, then we have screwed up
            # somewhere within this class.
            if isinstance(self._stochasticParam, dict):
                self.parameters = self._stochasticParam

        return self._integrate(self._odeTime, full_output)

    def integrate2(self, t, full_output=False, method=None):
        '''
        Integrate over a range of t when t is an array and a output
        at time t.  Select a suitable method to integrate when
        method is None.

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
        full_output: bool
            if we want additional information
        method: str, optional
            the integration method.  All those available in
            :class:`ode <scipy.integrate.ode>` are allowed with 'vode'
            and 'ivode' representing the non-stiff and stiff version
            respectively.  Defaults to None, which tries to choose the
            integration method via eigenvalue analysis (only one) using
            the initial conditions
        '''

        self._setIntegrateTime(t)
        # if our parameters are stochastic, then we are going to generate
        # another set of parameters to run
        if self._stochasticParam is not None:
            # this should always be true
            if isinstance(self._stochasticParam, dict):
                self.parameters = self._stochasticParam

        return self._integrate2(self._odeTime, full_output, method)

    def _setIntegrateTime(self, t):
        '''
        Set the full set of integration time including the origin
        '''

        assert self._t0 is not None, "Initial time not set"

        if ode_utils.is_list_like(t):
            if isinstance(t[0], Number):
                t = np.append(self._t0, t)
            else:
                raise ArrayError("Expecting a list of numeric value")
        elif isinstance(t, Number):
            t = np.append(self._t0, np.array(t))
        else:
            raise ArrayError("Expecting an array like input or a single " +
                             "numeric value")

        self._odeTime = t

    def _integrate(self, t, full_output=True):
        '''
        Integrate using :class:`scipy.integrate.odeint` underneath
        '''
        assert self._t0 is not None, "Initial time not set"

        f = ode_utils.integrate
        self._odeSolution, self._odeOutput = f(self,
                                               self._x0,
                                               t,
                                               full_output=True)
        if full_output:
            return self._odeSolution, self._odeOutput
        else:
            return self._odeSolution

    def _integrate2(self, t, full_output=True, method=None):
        '''
        Integrate using :class:`scipy.integrate.ode` underneath
        '''
        assert self._x0 is not None, "Initial state not set"

        f = ode_utils.integrateFuncJac
        self._odeSolution, self._odeOutput = f(self.ode_T,
                                               self.jacobian_T,
                                               self._x0,
                                               t[0], t[1::],
                                               includeOrigin=True,
                                               full_output=True,
                                               method=method)

        if full_output:
            return self._odeSolution, self._odeOutput
        else:
            return self._odeSolution

    def plot(self):
        '''
        Plot the results of the integration

        Notes
        -----
        If we have 3 states or more, it will always be arrange such
        that it has 3 columns.  Uses the operation from
        :mod:`odeutils`
        '''

        # just need to make sure that we have
        # already gotten the solution to the integration
        if self._odeSolution is None:
            try:
                self._integrate(self._odeTime)
                ode_utils.plot_det(self._odeSolution, self._odeTime, self._stateList)
            except:
                raise IntegrationError("Have not performed the integration yet")
        else:
            ode_utils.plot_det(self._odeSolution, self._odeTime, self._stateList)

    ########################################################################
    # Unrolling of the information from vector to sympy
    # t
    # state
    ########################################################################

    def _addTimeEvalParam(self, eval_param, t):
        eval_param.append((self._t, t))
        return eval_param

    def _addStateEvalParam(self, eval_param, state):
        super(DeterministicOde, self).state = state
        if self._state is not None:
            eval_param += self._state

        return eval_param

    def _getEvalParam(self, state, time, parameters):
        if state is None or time is None:
            raise InputError("Have to input both state and time")

        if parameters is not None:
            self.parameters = parameters
        elif self._parameters is None:
            if self.num_param == 0:
                pass
            else:
                raise InputError("Have not set the parameters yet")

        if isinstance(state, list):
            eval_param = state + [time]
        elif hasattr(state, '__iter__'):
            eval_param = list(state) + [time]
        else:
            eval_param = [state] + [time]

        return eval_param + self._paramValue
