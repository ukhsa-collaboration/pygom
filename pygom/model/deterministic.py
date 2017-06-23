"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    This module is defined such that operation on ode are all gathered
    in one place.  Future extension of operations should be added here

"""

__all__ = ['OperateOdeModel']

from .base_ode_model import BaseOdeModel
from ._model_errors import ArrayError, InputError, \
    IntegrationError, InitializeError
from ._model_verification import simplifyEquation
# import ode_utils as myUtil
# from .ode_utils import shapeAdjust, compileCode
from . import ode_utils
from . import _ode_composition

# import sympy.core.numbers
import sympy
from sympy.core.function import diff
import numpy
import scipy.linalg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import copy, io

class OperateOdeModel(BaseOdeModel):
    '''
    This contains the interface and operation
    built above the already defined set of ode

    Parameters
    ----------
    stateList: list
        A list of states (string)
    paramList: list
        A list of the parameters (string)
    derivedParamList: list
        A list of the derived parameters (tuple of (string,string))
    transitionList: list
        A list of transition (:class:`.Transition`)
    birthDeathList: list
        A list of birth or death process (:class:`.Transition`)
    odeList: list
        A list of ode (:class:`Transition`)

    '''

    def __init__(self,
                 stateList=None,
                 paramList=None,
                 derivedParamList=None,
                 transitionList=None,
                 birthDeathList=None,
                 odeList=None,
                 compileModule=None):
        '''
        Constructor that is built on top of a BaseOdeModel
        '''

        super(OperateOdeModel, self).__init__(stateList,
                                              paramList,
                                              derivedParamList,
                                              transitionList,
                                              birthDeathList,
                                              odeList)

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
        # because this is the Hessian of the ode which most of the
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
        self._SAUtil = ode_utils.shapeAdjust(self._numState, self._numParam)
        # compile the code.  Note that we need the class because we
        # compile both the formatted and unformatted version.
        self._SC = ode_utils.compileCode()

    def __eq__(self, other):
        if isinstance(other, OperateOdeModel):
            if self.getOde() == other.getOde():
                return True
            else:
                return False
        else:
            return False

    def __repr__(self):
        return "OperateOdeModel" + self._getModelStr() 

    ########################################################################
    #
    # Information about the ode
    #
    ########################################################################

    # TODO: check and see whether it is linear correctly!
    def isOdeLinear(self):
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
        isLinear = True
        # if we do not current possess the Jacobian, we find it! ROAR!
        if self._Jacobian is None:
            self.getJacobian()

        # a really stupid way to determining whether it is linear.
        # have not figured out a better way yet...
        a = self._Jacobian.atoms()
        for s in self._stateDict.values():
            if s in a:
                isLinear = False
#         for i in range(0, self._numState):
#             for j in range(0, self._numState):
#                 for k in range(0, self._numState):
#                     if self._Jacobian[i,j].has(self._stateList[k]):
#                         isLinear = False

        return isLinear

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

    def getOde(self, paramSub=False):
        '''
        Find the algebraic equations of the ode system.

        Returns
        -------
        :class:`sympy.matrices.matrices`
            ode in matrix form

        '''

        if self._ode is None:
            self._findOde()
        elif self._hasNewTransition:
            self._findOde()
        else:
            pass

        if paramSub:
            return self._ode.subs(self._parameters)
        else:
            return self._ode

    def printOde(self,latexOutput=False):

        A = self.getOde()
        B = sympy.zeros(A.rows,2)
        for i in range(A.shape[0]):
            B[i,0] = sympy.symbols('d' + str(self._stateList[i]) + '/dt=')
            B[i,1] = A[i]

        if latexOutput:
            print(sympy.latex(B, mat_str="array", mat_delim=None,
                              inv_trig_style='full'))
        else:
            sympy.pretty_print(B)

    def _findOde(self):
        self.getNumTransitions()
        # lets see how we have defined our ode
        # if it is explicit, then we go straight to the easy case
        if self._explicitOde:
            # we have explicit ode and we should obtain them directly
            super(OperateOdeModel, self)._computeOdeVector()
        else:
            # super(OperateOdeModel, self)._computeTransitionMatrix()
            # super(OperateOdeModel, self)._computeTransitionVector()
            # convert the transition matrix into the set of ode
            self._ode = sympy.zeros(self._numState, 1)
            pureTransitionList = self._getAllTransition(pureTransitions=True)
            fromList, \
                toList, \
                eqnList = self._unrollTransitionList(pureTransitionList)
            for i, eqn in enumerate(eqnList):
                for k in fromList[i]:
                    self._ode[k] -= eqn
                for k in toList[i]:
                    self._ode[k] += eqn

        # now we just need to add in the birth death processes
        super(OperateOdeModel, self)._computeBirthDeathVector()
        self._ode += self._birthDeathVector
        
        self._s = [s for s in self._iterStateList()] + [self._t]
        self._sp = self._s + [p for p in self._iterParamList()]
        # happy!
        self._hasNewTransition = False

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
        self._Grad = None
        self._Hessian = None
        self._Jacobian = None
        self._diffJacobian = None

        return self._ode
    
    def getTransitionGraph(self, fileName=None, show=True):
        '''
        Returns the transition graph using graphviz
        
        Parameters
        ----------
        fileName: str, optional
            name of the output file, defaults to None
        show: bool, optional
            If the graph should be plotted, defaults to True

        Returns
        -------
        dot:
            :class:`graphviz.Digraph`
        '''
        dot = _ode_composition.generateTransitionGraph(self, fileName)
        if show:
            img = mpimg.imread(io.BytesIO(dot.pipe("png")))
            plt.imshow(img)
            plt.show(block=False)
            return(dot)
        else:
            return(dot)
    
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
        return self.evalOde(time=t, state=state)

    def odeT(self, t, state):
        '''
        Same as :meth:`ode` but with t as the first parameter
        '''
        return self.ode(state, t)

    def evalOde(self, parameters=None, time=None, state=None):
        """
        Evaluate the ode given time, state and parameters.  An extension
        of :meth:`ode` but now also include the parameters.

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
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
        if self._ode is None or self._hasNewTransition:
            self.getOde()

        evalParam = self._getEvalParam(state, time, parameters)
        return self._odeCompile(evalParam)


    ########################################################################
    #
    # Jacobian related operations
    #
    ########################################################################

    def isStiff(self, state=None, t=None):
        '''
        Test on the eigenvalues of the Jacobian.  We classify the
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
        e = self.jacobianEigenvalue(state, t)
        return numpy.any(e > 0)

    def jacobianEigenvalue(self, state=None, t=None):
        '''
        Find out the eigenvalues of the Jacobian given state and time. If
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
                J = self.Jacobian(self._x0, self._t0)
        else:
            J = self.Jacobian(state, t)

        return scipy.linalg.eig(J)[0]


    def Jacobian(self, state, t):
        '''
        Evaluate the Jacobian given state and time

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
        return(self.evalJacobian(time=t, state=state))

    def JacobianT(self, t, state):
        '''
        Same as :meth:`Jacobian` but with t as first parameter
        '''
        return(self.Jacobian(state, t))

    def _Jacobian_NoCheck(self, state, t):
        return(self._evalJacobian_NoCheck(time=t, state=state))

    def _JacobianT_NoCheck(self, t, state):
        return(self._Jacobian_NoCheck(state, t))

    def getJacobian(self):
        '''
        Returns the Jacobian in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of state]

        '''
        if self._Jacobian is None:
            self.getOde()
            states = [s for s in self._iterStateList()]
            self._Jacobian = self._ode.jacobian(states)
            for i in range(self._numState):
                for j in range(self._numState):
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

        return(self._Jacobian)

    def evalJacobian(self, parameters=None, time=None, state=None):
        '''
        Evaluate the Jacobian given parameters, state and time. An extension
        of :meth:`.Jacobian` but now also include the parameters.

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
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
        :meth:`.Jacobian`

        '''
        if self._Jacobian is None or self._hasNewTransition:
            self.getOde()
            self.getJacobian()

        evalParam = self._getEvalParam(state, time, parameters)
        return(self._JacobianCompile(evalParam))

    def _evalJacobian_NoCheck(self, time, state):
        '''
        Same as :meth:`evalJacobian` but without the checks
        '''
        evalParam = list(state) + [time] + self._paramValue
        return(self._JacobianCompile(evalParam))

    ######  the sum of jacobian, i.e a_{i} = \sum_{j=1}^{d} J_{i,j}

    def SensJacobianState(self, stateParam, t):
        '''
        Evaluate the Jacobian of the sensitivity w.r.t. the
        state given state and time

        Parameters
        ----------
        stateParam: array like
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

        state = stateParam[0:self._numState]
        sens = stateParam[self._numState::]

        return(self.evalSensJacobianState(time=t, state=state, sens=sens))

    def SensJacobianStateT(self, t, state):
        '''
        Same as :meth:`SensJacobianStateT` but with t as first parameter
        '''
        return self.SensJacobianState(state, t)

    def evalSensJacobianState(self, parameters=None, time=None, state=None,
                              sens=None):
        '''
        Evaluate the Jacobian of the sensitivities w.r.t the states given
        parameters, state and time. An extension of :meth:`.SensJacobianState`
        but now also include the parameters.

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
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
        :meth:`.SensJacobianState`

        '''

        nS = self._numState
        nP = self._numParam

        # dot first, then transpose, then reshape
        # basically, some magic
        # don't ask me what is actually going on here, I did it
        # while having my wizard hat on
        return(numpy.reshape(self.diffJacobian(state, time).dot(
            self._SAUtil.vecToMatSens(sens)).transpose(), (nS*nP, nS)))

    ############################## derivative of Jacobian

    def diffJacobian(self, state, t):
        '''
        Evaluate the differential of Jacobian given state and time

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
        return(self.evalDiffJacobian(time=t, state=state))

    def diffJacobianT(self, t, state):
        '''
        Same as :meth:`diffJacobian` but with t as first parameter
        '''
        return(self.diffJacobian(state, t))

    def getDiffJacobian(self):
        '''
        Returns the Jacobian differentiate w.r.t. states in algebraic form

        Returns
        -------
        list
            list of size (num of state,) each with
            :mod:`sympy.matrices.matrices` of dimension
            [number of state x number of state]

        '''
        if self._diffJacobian is None:
            self.getOde()
            diffJac = list()
            
            for eqn in self._ode:
                J = sympy.zeros(self._numState, self._numState)
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

        return(self._diffJacobian)

    def evalDiffJacobian(self, parameters=None, time=None, state=None):
        '''
        Evaluate the differential of the Jacobian given parameters,
        state and time. An extension of :meth:`.diffJacobian` but now
        also include the parameters.

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
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
        :meth:`.Jacobian`

        '''
        if self._diffJacobian is None or self._hasNewTransition:
            self.getOde()
            self.getDiffJacobian()

        evalParam = self._getEvalParam(state, time, parameters)
        return(self._diffJacobianCompile(evalParam))

    ########################################################################
    #
    # Gradient related operations
    #
    ########################################################################

    def getGrad(self):
        '''
        Return the gradient of the ode in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of parameters]

        '''
        # finds
        
        if self._Grad is None:
            ode = self.getOde()
            self._Grad = sympy.zeros(self._numState, self._numParam)

            for i in range(self._numState):
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

        return(self._Grad)

    def Grad(self, state, time):
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
        return(self.evalGrad(state=state, time=time))

    def GradT(self, t, state):
        '''
        Same as :meth:`GradT` but with t as first parameter
        '''
        return(self.Grad(state, t))

    def evalGrad(self, parameters=None, time=None, state=None):
        '''
        Evaluate the gradient given parameters, state and time. An extension
        of :meth:`Grad` but now also include the parameters.

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
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
        :meth:`.Grad`

        '''
        if self._Grad is None or self._hasNewTransition:
            self.getOde()
            self.getGrad()

        evalParam = self._getEvalParam(state, time, parameters)
        return(self._GradCompile(evalParam))

    #
    # Jacobian of the Gradiant
    #

    def getGradJacobian(self):
        '''
        Return the Jacobian of the gradient in algebraic form

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state *
            number of parameters x number of state]

        See also
        --------
        :meth:`.getGrad`

        '''
        if self._GradJacobian is None:
            self._GradJacobian = sympy.zeros(self._numState*self._numParam,
                                             self._numState)
            G = self.getGrad()
            for k in range(0, self._numParam):
                for i in range(0, self._numState):
                    for j, s in enumerate(self._iterStateList()):
                        z = k*self._numState + i
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

        return(self._GradJacobian)

    def GradJacobian(self, state, time):
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
        :meth:`.Grad`

        """
        return(self.evalGradJacobian(state=state, time=time))

    def GradJacobianT(self, t, state):
        '''
        Same as :meth:`GradJacobian` but with t as first parameter
        '''
        return(self.GradJacobian(state, t))

    def evalGradJacobian(self, parameters=None, time=None, state=None):
        '''
        Evaluate the Jacobian of the gradient given parameters,
        state and time. An extension of :meth:`.GradJacobian`
        but now also include the parameters.

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
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
        :meth:`.GradJacobian`, :meth:`.getGradJacobian`

        '''
        if self._GradJacobian is None or self._hasNewTransition:
            self.getOde()
            self.getGradJacobian()           

        evalParam = self._getEvalParam(state, time, parameters)
        return self._GradJacobianCompile(evalParam)

    ########################################################################
    #
    # Hessian related operations
    #
    ########################################################################

    def getHessian(self):
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
            ode = self.getOde()
            self._Hessian = list()
            # roll out the equation one by one.  Each H below is a the
            # second derivative of f_{j}(x), the j^{th} ode.  Each ode
            # correspond to a state
            for eqn in ode:
                H = sympy.zeros(self._numParam, self._numParam)
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

    def Hessian(self, state, time):
        """
        Evaluate the Hessian given state and time

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
        A = self.evalHessian(state=state, time=time)
        return [numpy.array(H, float) for H in A]

    def evalHessian(self, parameters=None, time=None, state=None):
        '''
        Evaluate the Hessian given parameters, state and time. An extension
        of :meth:`Hessian` but now also include the parameters.

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
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
        :meth:`.Grad`, :meth:`.evalGrad`

        '''
        if self._hasNewTransition:
            self.getOde()

        evalParam = list()
        evalParam = self._addTimeEvalParam(evalParam, time)
        evalParam = self._addStateEvalParam(evalParam, state)

        if parameters is None:
            if self._HessianWithParam is None:
                self._computeHessianParam()
        else:
            self.setParameters(parameters)

        if self._Hessian is None:
            self._computeHessianParam()

        if len(evalParam) == 0:
            return self._Hessian
        else:
            H = list()
            for i in range(0, self._numState):
                H = self._HessianWithParam[i].subs(evalParam)
            return H

    def _computeHessianParam(self):
        self._Hessian = self.getHessian()

        self._HessianWithParam = copy.deepcopy(self._Hessian)
        for H in self._HessianWithParam:
            H = H.subs(self._parameters)

        return None

    ########################################################################
    #
    # Sensitivity related operations (1st forward)
    #
    ########################################################################

    def sensitivity(self, sens, t, state, byState=False):
        """
        Evaluate the sensitivity given state and time.  The default is to
        output the values by parameters, i.e. :math:`s_{i},\ldots,s_{i+n}` are
        partial derivatives w.r.t. the states for :math:`i \in {1,1+p,1+2p,1+3p
        \ldots, 1+(n-1)p}`.  This is to take advantage of the fact that
        we have a block diagonal Jacobian that was already evaluated

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
        byState: bool
            how we want the output to be arranged.  Default is True so
            that we have a block diagonal structure

        Returns
        -------
        :class:`numpy.ndarray`
            f(s)

        """
        # TODO: allows the functionality to not evaluate all sensitivity

        # S = \nabla_{time} \frac{\partial State}{\partial Parameters}
        # rearrange the input if required
        if byState:
            S = numpy.reshape(sens, (self._numState, self._numParam))
        else:
            S = self._SAUtil.vecToMatSens(sens)

        return self.evalSensitivity(S=S, t=t, state=state, byState=byState)

    def sensitivityT(self, t, sens, state, byState=False):
        '''
        Same as :meth:`sensitivity` but with t as first parameter
        '''
        return self.sensitivity(sens, t, state, byState)

    def evalSensitivity(self, S, t, state, byState=False):
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
        byState: bool
            how we want the output to be arranged.  Default is True so
            that we have a block diagonal structure

        Returns
        -------
        :class:`numpy.ndarray`
            f(s)

        Notes
        -----
        It is different to :meth:`.evalOde` and :meth:`.evalJacobian` in that
        the extra input argument is not a parameter

        See Also
        --------
        :meth:`.sensitivity`

        """

        # Jacobian * sensitivities + G
        # where G is the gradient
        J = self.Jacobian(state, t)
        G = self.Grad(state, t)
        A = numpy.dot(J, S) + G
        
        if byState:
            return numpy.reshape(A, self._numState*self._numParam)
        else:
            return self._SAUtil.matToVecSens(A)

    def odeAndSensitivity(self, stateParam, t, byState=False):
        '''
        Evaluate the sensitivity given state and time

        Parameters
        ----------
        stateParam: array like
            The current numerical value for the states as well as the
            sensitivities values all in one.  We assume that the state
            values comes first.
        t: double
            The current time
        byState: bool
            Whether the output vector should be arranged by state or by
            parameters. If False, then it means that the vector of output is
            arranged according to looping i,j from Sensitivity_{i,j} with i the
            state and j the param. This is the preferred way because it leads
            to a block diagonal Jacobian

        Returns
        -------
        :class:`list`
            concatenation of 2 element. First contains the ode, second the
            sensitivity. Both are of type :class:`numpy.ndarray`

        See Also
        --------
        :meth:`.sensitivity` , :meth:`.ode`

        '''

        if len(stateParam) == self._numState:
            raise InputError("You have only inputed the initial condition " +
                             "for the states and not the sensitivity")

        # unrolling, assuming that we would always put the state first
        # there is no safety checks on this because it is impossible to
        # distinguish what is state and what is sensitivity as they are
        # all numeric value that can take the full range (-\infty,\infty)
        state = stateParam[0:self._numState]
        sens = stateParam[self._numState::]

        out1 = self.ode(state, t)
        out2 = self.sensitivity(sens, t, state, byState)
        return numpy.append(out1, out2)

    def odeAndSensitivityT(self, t, stateParam, byState=False):
        '''
        Same as :meth:`odeAndSensitivity` but with t as first parameter
        '''
        return self.odeAndSensitivity(stateParam, t, byState)

    def odeAndSensitivityJacobian(self, stateParam, t, byState=False):
        '''
        Evaluate the sensitivity given state and time.  Output a block
        diagonal sparse matrix as default.

        Parameters
        ----------
        stateParam: array like
            The current numerical value for the states as well as the
            sensitivities values all in one.  We assume that the state
            values comes first.
        t: double
            The current time
        byState: bool
            How the output is arranged, according to the vector of output.
            It can be in terms of state or parameters, where by state means
            that the Jacobian is a block diagonal matrix.

        Returns
        -------
        :class:`numpy.ndarray`
            output of a square matrix of size: number of ode + 1 times number
            of parameters

        See Also
        --------
        :meth:`.odeAndSensitivity`

        '''

        if len(stateParam) == self._numState:
            raise InputError("Expecting both the state and the sensitivities")
        else:
            state = stateParam[0:self._numState]

        # now we start the computation
        J = self.Jacobian(state, t)
        # create the block diagonal Jacobian, assuming that whoever is
        # calling this function wants it arranges by state-parameters

        # Note that none of the ode integrator in scipy allow a sparse Jacobian
        # matrix.  All of them accept a banded matrix in packed format but not
        # an actual sparse, or specifying the number of bands.
        outJ = numpy.kron(numpy.eye(self._numParam), J)
        # jacobian of the gradient
        GJ = self.GradJacobian(state, t)
        # and now we add the gradient
        sensJacobianOfState = GJ + self.SensJacobianState(stateParam, t)

        if byState:
            arrangeVector = numpy.zeros(self._numState * self._numParam)
            k = 0
            for j in range(0, self._numParam):
                for i in range(0, self._numState):
                    if i == 0:
                        arrangeVector[k] = (i*self._numState) + j
                    else:
                        arrangeVector[k] = (i*(self._numState - 1)) + j
                    k += 1

            outJ = outJ[numpy.array(arrangeVector,int),:]
            idx = numpy.array(arrangeVector, int)
            sensJacobianOfState = sensJacobianOfState[idx,:]
        # The Jacobian of the ode, then the sensitivities w.r.t state and
        # the sensitivities. In block form.  Theoreticaly, only the diagonal
        # blocks are important but we output the full matrix for completeness
        return numpy.asarray(numpy.bmat([
            [J, numpy.zeros((self._numState, self._numState*self._numParam))],
            [sensJacobianOfState, outJ]
        ]))

    def odeAndSensitivityJacobianT(self, t, stateParam, byState=False):
        '''
        Same as :meth:`odeAndSensitivityJacobian` but with t as first parameter
        '''
        return self.odeAndSensitivityJacobian(stateParam, t, byState)

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
        output the values by parameters, i.e. :math:`s_{i},\ldots,s_{i+n}` are
        partial derivatives w.r.t. the states for :math:`i \in {1,1+p,1+2p,1+3p
        \ldots, 1+(n-1)p}`.  This is to take advantage of the fact that
        we have a block diagonal Jacobian that was already evaluated

        Parameters
        ----------
        sensIV: array like
            The starting sensitivity of size [number of state x number of
            parameters] + [number of state x number of state] for the
            initial condition.  The latter is an identity matrix at time
            zero.
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

        nS = self._numState
        nP = self._numParam
        # separate information out.  Again, we do have not have checks here
        # as it will be impossible to distinguish what is correct
        sens = sensIV[:(nS*nP)]
        S = self._SAUtil.vecToMatSens(sens)

        IV = numpy.reshape(sensIV[-(nS*nS):], (nS, nS), 'F')

        return self.evalSensitivityIV(S=S, IV=IV, t=t, state=state)

    def sensitivityIVT(self, t, sens, IV, state):
        '''
        Same as :meth:`sensitivityIV` but with t as first parameter
        '''
        return self.sensitivityIV(sens, IV, t, state)

    def evalSensitivityIV(self, S, IV, t, state):
        """
        Evaluate the sensitivity with initial values given
        state and time

        Parameters
        ----------
        S: array like
            Which should be :class:`numpy.ndarray`.
            The starting sensitivity of size [number of state x number of
            parameters].  Which are normallly zero or one,
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
        It is different to :meth:`.evalOde` and :meth:`.evalJacobian` in that
        the extra input argument is not a parameter

        See Also
        --------
        :meth:`.sensitivityIV`

        """
        # Jacobian * sensitivities + G
        # where G is the gradient
        # Evidently, A below uses the same operations as
        # A = self.evalSensitivity(S,t,state)
        # but we are evaluating them explicitly here because
        # we will be using J as well when computing B

        J = self.Jacobian(state, t)
        G = self.Grad(state, t)
        A = numpy.dot(J, S) + G

        # and Jacobian * sensitivities of the initial condition
        B = numpy.dot(J, IV)

        # we want to output by parameters
        return self._SAUtil.matToVecSens(A), B.flatten('F')

    def odeAndSensitivityIV(self, stateParam, t):
        '''
        Evaluate the sensitivity given state and time

        Parameters
        ----------
        stateParam: array like
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
        :meth:`.sensitivity` , :meth:`.ode`

        '''

        if len(stateParam) == self._numState:
            raise InputError("You have only inputed the initial condition " +
                             "for the states and not the sensitivity")

        # unrolling, assuming that we would always put the state first
        state = stateParam[0:self._numState]
        # the remainings
        sensIV = stateParam[self._numState::]
        # separate evaluation
        out1 = self.ode(state,t)
        out2,out3 = self.sensitivityIV(sensIV, t, state)
        return numpy.append(numpy.append(out1, out2), out3)

    def odeAndSensitivityIVT(self, t, stateParam):
        '''
        Same as :meth:`odeAndSensitivityIV` but with t as first parameter
        '''
        return self.odeAndSensitivityIV(stateParam, t)

    def odeAndSensitivityIVJacobian(self, stateParam, t):
        '''
        Evaluate the sensitivity given state and time.  Output a block
        diagonal sparse matrix as default.

        Parameters
        ----------
        stateParam: array like
            The current numerical value for the states as well as the
            sensitivities values all in one.  We assume that the state
            values comes first.
        t: double
            The current time
        byState: bool
            How the output is arranged, according to the vector of output.
            It can be in terms of state or parameters, where by state means
            that the Jacobian is a block diagonal matrix.

        Returns
        -------
        :class:`numpy.ndarray`
            output of a square matrix of size: number of ode + 1 times number
            of parameters

        See Also
        --------
        :meth:`.odeAndSensitivity`

        '''

        if len(stateParam) == self._numState:
            raise InputError("Expecting both the state and the sensitivities")
        else:
            state = stateParam[0:self._numState]

        nS = self._numState
        nP = self._numParam
        # now we start the computation, the simply one :)
        J = self.Jacobian(state, t)

        # now the Jacobian of the state vs initial value
        DJ = self.diffJacobian(state, t)
        A = DJ.dot(numpy.reshape(stateParam[(nS*(nP+1))::], (nS, nS), 'F'))
        A = numpy.reshape(A.transpose(), (nS*nS, nS))
        
        if nP == 0:
            return numpy.asarray(numpy.bmat([
                        [J, numpy.zeros((nS, nS*nS))],
                        [A, numpy.kron(numpy.eye(nS), J)]
                        ]))
        else:

            # create the block diagonal Jacobian, assuming that whoever is
            # calling this function wants it arranges by state-parameters
            outJ = numpy.kron(numpy.eye(nP), J)

            # jacobian of the gradient
            GJ = self.GradJacobian(state, t)
            GS = self.SensJacobianState(stateParam[:(nS*(nP + 1))], t)
            sensJacobianOfState = GJ + GS

            # The Jacobian of the ode, then the sensitivities w.r.t state 
            # and the sensitivities. In block form
            return numpy.asarray(numpy.bmat([
                [J, numpy.zeros((nS, nS*nP)), numpy.zeros((nS, nS*nS))],
                [sensJacobianOfState, outJ, numpy.zeros((nS*nP, nS*nS))],
                [A, numpy.zeros((nS*nS, nS*nP)), numpy.kron(numpy.eye(nS), J)]
            ]))

    def odeAndSensitivityIVJacobianT(self, t, stateParam):
        '''
        Same as :meth:`odeAndSensitivityIVJacobian` but with t as first parameter
        '''
        return self.odeAndSensitivityIVJacobian(stateParam, t)

    ############################################################################
    #
    # Adjoint
    #
    ############################################################################

    def adjointInterpolate(self, state, t, interpolateFuncList, objInput=None):
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
        interpolateFuncList: list
            list of interpolating functions of the state
        objInput: callable
            This should take inputs similar to an ode, i.e. of the form
            func(y,t).  If j(y,t) is the cost function, then objInput
            is a function that calculates :math:`\partial j \over \partial x`.

        Returns
        -------
        :class:`numpy.ndarray`
            output of the same length as the ode
        '''
        stateParam = [o(t) for o in interpolateFuncList]
        return self.adjoint(state, t, stateParam, objInput)

    def adjointInterpolateT(self, t, state, interpolateFuncList, objInput=None):
        '''
        Same as :meth:`adjointInterpolate` but with t as first parameter
        '''
        return self.adjointInterpolate(state, t, interpolateFuncList, objInput)

    def _adjointInterpolate_NoCheck(self, state, t,
                                    interpolateFuncList, objInput=None):
        stateParam = [o(t) for o in interpolateFuncList]
        return self._adjoint_NoCheck(state, t, stateParam, objInput)

    def _adjointInterpolateT_NoCheck(self, t, state, 
                                     interpolateFuncList, objInput=None):
        return self._adjoint_NoCheck(state, t, interpolateFuncList, objInput)

    def adjoint(self, state, t, stateParam, objInput=None):
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
        stateParam: array like
            The state vector that is (or maybe) required to evaluate the
            Jacobian of the original system
        objInput: callable
            This should take inputs similar to an ode, i.e. of the form
            func(y,t).  If j(y,t) is the cost function, then objInput
            is a function that calculates :math:`\partial j \over \partial x`.

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
        J = self.Jacobian(stateParam, t)

        if objInput is None:
            return numpy.dot(state, -J)
        else:
            return objInput(stateParam, t) - J.transpose().dot(state)

    def _adjoint_NoCheck(self, state, t, stateParam, objInput=None):
        J = self._Jacobian_NoCheck(stateParam, t)
        if objInput is None:
            return numpy.dot(state, -J)
        else:
            return objInput(stateParam, t) - J.transpose().dot(state)

    def _adjoinT_NoCheck(self, t, state, stateParam, objInput=None):
        return self._adjointT_NoCheck(state, t, stateParam, objInput)

    def adjointT(self, t, state, stateParam, objInput=None):
        '''
        Same as :meth:`adjoint` but with t as first parameter
        '''
        return self.adjoint(state, t, stateParam, objInput)

    def adjointJacobian(self, state, t, stateParam, objInput=None):
        '''
        Compute the Jacobian of the adjoint given the adjoint vector, time,
        state variable and the objective function.  This is simply the same
        as the negative Jacobian of the ode transposed.

        Parameters
        ----------
        state: array like
            The current value of lambda, where lambda's are the Lagrangian
            multipliers of the differential equation.
        t: double
            The current time.
        stateParam: array like
            The state vector that is (or maybe) required to evaluate the
            Jacobian of the original system
        objInput: callable
            This should take inputs similar to an ode, i.e. of the form
            func(y,t).  If j(y,t) is the cost function, then objInput
            is a function that calculates :math:`\partial j \over \partial x`.

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
        return -self.Jacobian(stateParam, t).transpose()

    def adjointJacobianT(self, t, state, stateParam, objInput=None):
        '''
        Same as :meth:`adjointJacobianT` but with t being the
        first parameter
        '''
        return self.adjointJacobian(state, t, stateParam, objInput)

    def adjointInterpolateJacobian(self, state, t,
                                   interpolateFuncList, objInput=None):
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
        interpolateFuncList: list
            list of interpolating functions of the state
        objInput: callable
            This should take inputs similar to an ode, i.e. of the form
            func(y,t).  If j(y,t) is the cost function, then objInput
            is a function that calculates :math:`\partial j \over \partial x`.

        Returns
        -------
        :class:`numpy.ndarray`
            output of is a two dimensional array of size
            [number of state x number of state]

        Notes
        -----
        Same as :meth:`.adjointJacobian` but takes a list of interpolating
        function instead of a single (vector) value

        See Also
        --------
        :meth:`.adjointJacobian`

        '''
        stateParam = [o(t) for o in interpolateFuncList]
        return self.adjointJacobian(state, t, stateParam, objInput)

    def adjointInterpolateJacobianT(self, t, state,
                                    interpolateFuncList, objInput=None):
        '''
        Same as :meth:`adjointInterpolateJacobian` but with t as first parameter
        '''
        return self.adjointInterpolateJacobian(state, t,
                                               interpolateFuncList, objInput)

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
        return self.evalForwardforward(FF=FF, S=S, state=state, t=t)

    def forwardforwardT(self, t, ff, s, state):
        '''
        Same as :meth:`forwardforward` but with t as the first
        parameter
        '''
        return self.forwardforward(ff, t, state, s)

    def evalForwardforward(self, FF, S, state, t):
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

        J = self.Jacobian(state, t)
        diffJ = self.diffJacobian(state, t)

        # evaluating by state/ode, the matrix of second derivative
        # we have kron products into all these evaluations and the class
        # here use a sparse matrix operation
        outFF = self._SAUtil.kronParam(J).dot(FF)
        outFF += self._SAUtil.kronState(A=S.T, pre=True).dot(diffJ).dot(S)

        # now we need to magic our list / matrix into a vector, aka append
        # each of the vectorized matrix one after another
        return self._SAUtil.matToVecFF(outFF)

    def odeAndForwardforward(self, stateParam, t):
        '''
        Evaluate a single f(x) of the ode and the
        forward-forward sensitivities

        Parameters
        ----------
        stateParam: array like
            state and forward-forward sensitivities in vector form
        t: numeric
            time

        Returns
        -------
        :class:`numpy.ndarray`
            same size as the stateParam input
        '''
        
        if len(stateParam) == self._numState:
            raise InputError("You have only inputed the initial condition " +
                             "for the states and not the sensitivity")
        elif len(stateParam) == ((self._numState + 1)*self._numParam):
            raise InputError("You have only inputed the initial condition " +
                             "for the states and the sensitivity but not the " +
                             "forward forward condition")

        # unrolling of parameters
        state = stateParam[0:self._numState]
        # we want the index up to numState * (numParam + 1)
        # as in, (numState * numParam + numState,
        # number of sensitivities + number of ode)
        sens = stateParam[self._numState:(self._numState*(self._numParam + 1))]
        # the rest are then the forward forward sensitivities
        ff = stateParam[(self._numState*(self._numParam + 1))::]

        out1 = self.ode(state, t)
        out2 = self.sensitivity(sens, t, state)
        out3 = self.forwardforward(ff, t, state, sens)

        return numpy.append(numpy.append(out1, out2), out3)

    def odeAndForwardforwardT(self, t, stateParam):
        '''
        Same as :meth:`odeAndForwardForward` but with time
        as the first input

        '''
        return self.odeAndForwardforward(stateParam, t)

    def odeAndForwardforwardJacobian(self, stateParam, t):
        '''
        Return the Jacobian after evaluation given the input
        of the state and the forward forward sensitivities

        Parameters
        ----------
        stateParam: array like
            state and forward-forward sensitivities in vector form
        t: numeric
            time

        Returns
        -------
        :class:`numpy.ndarray`
            size of (a,a) where a is the length of the
            stateParam input
        '''
        if len(stateParam) == self._numState:
            state = stateParam
        else:
            state = stateParam[0:self._numState]

        J = self.Jacobian(state, t)
        # create the block diagonal Jacobian, assuming that whoever is
        # calling this function wants it arranges by state-parameters
        # We are only construct the block diagonal Jacobian here
        # instead of the full one unlike some of the other methods within
        # this class
        outJS = numpy.kron(numpy.eye(self._numParam), J)
        outJFF = numpy.kron(numpy.eye(self._numParam*self._numParam), J)
        # The Jacobian of the ode, then the sensitivities, then the
        # forward forward sensitivities
        return scipy.linalg.block_diag(J, outJS, outJFF)

    def odeAndForwardforwardJacobianT(self, t, stateParam):
        '''
        Same as :meth:`odeAndForwardforwardJacobian` but
        with t being the first parameters
        '''
        return self.odeAndForwardforwardJacobian(stateParam, t)

    ########################################################################
    #
    # Initial conditions, integrations and result plots
    #
    ########################################################################

    def setInitialState(self, x0):
        '''
        Set the initial state values

        Parameters
        ----------
        x0: array like
             initial condition of x at time 0

        '''
        err_str = "More than one state in the defined system"

        if isinstance(x0, numpy.ndarray):
            self._x0 = x0
        elif isinstance(x0, (list, tuple)):
            self._x0 = numpy.array(x0)
        elif isinstance(x0, (int, float)):
            if self._numState == 1:
                self._x0 = numpy.array([x0])
            else:
                raise InitializeError(err_str)
        else:
            raise InitializeError("err_str")

        if len(self._x0) != self._numState:
            raise Exception("Number of state is " +
                            str(self._numState)+ " but " +
                            str(len(self._x0))+ " detected")

        return self

    def setInitialTime(self, t0):
        '''
        Set the initial time

        Parameters
        ----------
        t0: numeric
            initial time where x0 is observed

        '''

        err_str = "Initial time should be a "
        if ode_utils.isNumeric(t0):
            self._t0 = t0
        elif ode_utils.isListLike(t0):
            if len(t0) == 1:
                if ode_utils.isNumeric(t0[0]):
                    self._t0 = t0[0]
                else:
                    raise InitializeError(err_str + "numeric value")
            else:
                raise InitializeError(err_str + "single value")
        elif isinstance(t0, (list, tuple)):
            if len(t0) == 1:
                self._t0 = numpy.array(t0[0])
            else:
                raise InitializeError(err_str + "single value")
        else:
            raise InitializeError(err_str + "numeric value")

        return self

    def setInitialValue(self, x0, t0):
        '''
        Set the initial values, both time and state

        Parameters
        ----------
        x0: array like
            initial condition of x at time 0
        t0: numeric
            initial time where x0 is observed

        '''
        self.setInitialState(x0)
        self.setInitialTime(t0)
        return self

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
                self.setParameters(self._stochasticParam)

        return self._integrate(self._odeTime, full_output)

    def integrate2(self, t, full_output=False, intName=None):
        '''
        Integrate over a range of t when t is an array and a output
        at time t.  Select a suitable method to integrate when
        intName is None.

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
        full_output: bool
            if we want additional information
        intName: str, optional
            the integration method.  All those availble in
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
                self.setParameters(self._stochasticParam)

        return self._integrate2(self._odeTime,full_output,intName)

    def _setIntegrateTime(self, t):
        '''
        Set the full set of integration time including the origin
        '''
        
        assert self._t0 is not None, "Initial time not set"

        if ode_utils.isListLike(t):
            if ode_utils.isNumeric(t[0]):
                t = numpy.append(self._t0, t)
            else:
                raise ArrayError("Expecting a list of numeric value")
        elif ode_utils.isNumeric(t):
            t = numpy.append(self._t0, numpy.array(t))
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

    def _integrate2(self, t, full_output=True, intName=None):
        '''
        Integrate using :class:`scipy.integrate.ode` underneath
        '''
        assert self._x0 is not None, "Initial state not set"

        f = ode_utils.integrateFuncJac
        self._odeSolution, self._odeOutput = f(self.odeT,
                                               self.JacobianT,
                                               self._x0,
                                               t[0], t[1::],
                                               includeOrigin=True,
                                               full_output=True,
                                               intName=intName)

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
                ode_utils.plot(self._odeSolution, self._odeTime, self._stateList)
            except:
                raise IntegrationError("Have not performed the integration yet")
        else:
            ode_utils.plot(self._odeSolution, self._odeTime, self._stateList)

    ########################################################################
    # Unrolling of the information from vector to sympy
    # t
    # state
    ########################################################################

    def _addTimeEvalParam(self, evalParam, t):
        evalParam.append((self._t, t))
        return evalParam

    def _addStateEvalParam(self, evalParam, state):
        super(OperateOdeModel, self).setState(state)
        if self._state is not None:
            evalParam += self._state

        return evalParam
    
    def _getEvalParam(self, state, time, parameters):
        if state is None or time is None:
            raise InputError("Have to input both state and time")
        
        if parameters is not None:
            self.setParameters(parameters)
        elif self._parameters is None:
            if len(self._paramList) == 0:
                pass
            else:
                raise InputError("Have not set the parameters yet")

        if isinstance(state, list):
            evalParam = state + [time]
        elif hasattr(state, '__iter__'):
            evalParam = list(state) + [time]
        else:
            evalParam = [state] + [time]

        return evalParam + self._paramValue
