"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module/class that carries out different type of simulation
    on an ode formulation

"""

__all__ = ['SimulateOdeModel']

from .deterministic import OperateOdeModel
from .stochastic_simulation import firstReaction, tauLeap
from .transition import TransitionType, Transition
from ._model_errors import InputError, SimulationError
from ._model_verification import checkEquation, simplifyEquation
from . import _ode_composition
from . import ode_utils

import numpy
import sympy
import scipy.stats
import copy

class SimulateOdeModel(OperateOdeModel):
    '''
    This builds on top of :class:`OperateOdeModel` which we
    simulate the outcome instead of solving it deterministically

    Parameters
    ----------
    stateList: list
        A list of states (string)
    paramList: list
        A list of the parameters (string)
    derivedParamList: list
        A list of the derived parameters (tuple of (string,string))
    transitionList: list
        A list of transition (:class:`Transition`)
    birthDeathList: list
        A list of birth or death process (:class:`Transition`)
    odeList: list
        A list of ode (:class:`Transition`)

    '''

    def __init__(self,
                 stateList=None,
                 paramList=None,
                 derivedParamList=None,
                 transitionList=None,
                 birthDeathList=None,
                 odeList=None):
        '''
        Constructor that is built on top of OperateOdeModel
        '''

        super(SimulateOdeModel, self).__init__(stateList,
                                               paramList,
                                               derivedParamList,
                                               transitionList,
                                               birthDeathList,
                                               odeList)

        # need a manual override because it is possible that we
        # want to perform simulation in a parallel/distributed manner
        # and there are issues with pickling fortran objects
        self._SC = ode_utils.compileCode(backend='cython')

        self._birthDeathRate = None
        self._birthDeathRateCompile = None

        # information required for the tau leap
        self._transitionJacobian = None

        self._transitionMean = None
        self._transitionMeanCompile = None

        self._transitionVar = None
        self._transitionVarCompile = None

        self._transitionVectorCompile = None
        self._transitionMatrixCompile = None
        
        # micro times for jumps
        self._tau = None
        self._tauDict = None

    def __repr__(self):
        return "SimulateOdeModel" + self._getModelStr()

    def simulateParam(self, t, iteration, full_output=False):
        '''
        Simulate the ode by generating new realization of the stochastic
        parameters and integrate the system deterministically.

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
        iteration: int
            number of iterations you wish to simulate
        full_output: bool, optional
            if we want additional information, Yall in the return,
            defaults o false

        Returns
        -------
        Y: :class:`numpy.ndarray`
            of shape (len(t),len(state)), mean of all the simulation
        Yall: :class:`numpy.ndarray`
            of shape (iteration,len(t),len(state))
        '''

        # if our parameters not stochastic, then we are going to
        # throw a warning because simulating a deterministic system is
        # just plain stupid
        if self._stochasticParam is None:
            raise InputError("Deterministic parameters.")
        if iteration is None:
            raise InputError("Need to specify the number of iterations")
        if t is None:
            raise InputError("Need to specify the time we wish to observe")

        self._odeSolution = self.integrate(t)

        # try to compute the simulation in parallel
        try:
            for i in self._stochasticParam:
                if isinstance(i, scipy.stats._distn_infrastructure.rv_frozen):
                    raise Exception("Cannot perform parallel simulation "
                                   +"using a serialized object as distribution")
            # check the type of parameter we have as input
            import dask.bag
            y = list()
            for i in range(iteration):
                y_i = list()
                for key, rv in self._stochasticParam.items():
                    y_i += [{key:rv.rvs(1)[0]}]
                y += [y_i]  
            # y = [rv.rvs(iteration) for rv in self._stochasticParam.values()]
            # y = numpy.array(list(zip(*y)))
            def sim(x):
                self.setParameters(x)
                return(self.integrate(t))

            # def sim(t1): return(self.integrate(t1))

            # xtmp = dask.bag.from_sequence([t]*iteration)
            xtmp = dask.bag.from_sequence(y)
            solutionList = xtmp.map(sim).compute()
        except Exception: # as e:
            # print(e)
            # print("Serial")
            solutionList = [self.integrate(t) for i in range(iteration)]

        # now make our 3D array
        # the first dimension is the number of iteration
        Y = numpy.dstack(solutionList).mean(axis=2)

        if full_output:
            return Y, solutionList
        else:
            return Y

    def simulateJump(self, t, iteration, exact=False, full_output=False):
        '''
        Simulate the ode using stochastic simulation.  It switches
        between a first reaction method and a :math:`\\tau`-leap
        algorithm internally.

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
            or the final time point
        iteration: int
            number of iterations you wish to simulate
        exact: bool, optional
            True if exact simulation is desired, defaults to False
        full_output: bool,optional
            if we want additional information, simT

        Returns
        -------
        simX: list
            of length iteration each with (len(t),len(state)) if t is a vector,
            else it outputs unequal shape that was record of all the jumps
        simT: list or :class:`numpy.ndarray`
            if t is a single value, it outputs unequal shape that was
            record of all the jumps.  if t is a vector, it outputs t so that
            it is a :class:`numpy.ndarray` instead

        '''

        assert len(self._odeList) == 0, \
            "Currently only able to simulate when only transitions are present"
        assert numpy.all(numpy.mod(self._x0, 1) == 0), \
            "Can only simulate a jump process with integer initial values"
        
        # this determines what type of output we want
        timePoint = False

        if ode_utils.isNumeric(t):#, (int, float, numpy.int64, numpy.float64)):
            finalT = t
        elif isinstance(t, (list, tuple)):
            t = numpy.array(t)
            if len(t) == 1:
                finalT = t
            else:
                finalT = t[-1:]
                timePoint = True
        elif isinstance(t, numpy.ndarray):
            finalT = t[-1:]
            timePoint = True
        else:
            raise InputError("Unknown data type for time")

        if self._transitionVectorCompile is None:
            self._compileTransitionVector()

        try:
            import dask.bag
            def jump_partial(final_t): return(self._jump(final_t,
                                                         exact=exact,
                                                         full_output=True,
                                                         seed=True))

            xtmp = dask.bag.from_sequence(numpy.ones(iteration)*finalT)
            xtmp = xtmp.map(jump_partial).compute()
        except Exception:# as e:
            # print(e)
            xtmp = [self._jump(finalT, exact=exact, full_output=True) for _i in range(iteration)]
        
        xmat = list(zip(*xtmp))
        simXList, simTList = list(xmat[0]), list(xmat[1])
        print("Finish computation")
        # now we want to fix our simulation, if they need fixing that is
        # print timePoint
        if timePoint:
            for _i in range(len(simXList)):
                # unroll, always the first element
                # it is easy to remember that we are accessing the first
                # element because pop is spelt similar to poop and we
                # all know that your execute first in first out when you
                # poop!
                simX = simXList.pop(0)
                simT = simTList.pop(0)

                x = self._extractObservationAtTime(simX, simT, t)
                simTList.append(t)
                simXList.append(x)
        # note that we have to remain in list form because the number of
        # simulation will be different if we are not dealing with
        # a specific set of time points

        if full_output:
            if timePoint:
                return simXList, t
            else:
                return simXList, simTList
        else:
            return simXList

    def _extractObservationAtTime(self, X, t, targetTime):
        '''
        Given simulation and a set of time points which we would like to
        observe, we extract the observations :math:`x_{t}` with
        :math:`\\min\\{ \\abs( t - targetTime) \\}`
        '''
        y = list()
        # maxTime = max(t)
        index = 0
        for i in targetTime:
            if numpy.any(t == i):
                index = numpy.where(t == i)[0][0]
            else:
                index = numpy.searchsorted(t, i) - 1
            y.append(X[index])

        return numpy.array(y)

    def _jump(self, finalT, exact=False, full_output=True, seed=None):
        '''
        Jumps from the initial time self._t0 to the input time finalT
        '''
        # initial time
        assert self._t0 is not None, "No initial time"
        assert self._x0 is not None, "No initial state"

        if seed: seed = numpy.random.RandomState()
        t = self._t0.tolist()
        x = copy.deepcopy(self._x0)

        # holders and record information
        xList = [x.copy()]
        tList = [t]

        # we want to construct some jump times
        if self._GMat is None:
            self._computeDependencyMatrix()

#         rates = self.transitionVector(x,t)
#         jumpTimes = numpy.array([self._t0 + rexp(1, r) for r in rates])
#         print rates
#         print jumpTimes 
        # keep jumping, Whoop Whoop (put your hands up!).
        f = firstReaction
        while t < finalT:
            try:
                if exact:
                    x, t, success = f(x, t, self._vMat,
                                      self.transitionVector, seed=seed)
                else:
                    if numpy.min(x) > 10:
                        x_tmp, t_tmp, success = tauLeap(x, t,
                                                self._vMat, self._lambdaMat,
                                                self.transitionVector,
                                                self.transitionMean,
                                                self.transitionVar,
                                                seed=seed)
                        if success:
                            x, t = x_tmp, t_tmp
                        else:
                            x, t, success = f(x, t, self._vMat,
                                              self.transitionVector, seed=seed)                            
                    else:
                        x, t, success = f(x, t, self._vMat,
                                          self.transitionVector, seed=seed)
## print("Directly into the firstReaction method and is it good? %s" % success) 
                if success:
                    xList.append(x.copy())
                    tList.append(t)
                else:
                    break
                    ## print("x: %s, t: %s" % (x, t))
                    ## raise Exception('WTF')
            except SimulationError:
                break
        return numpy.array(xList), numpy.array(tList)

    ########################################################################
    #
    # Operators with regard to the transition matrix
    #
    ########################################################################

    def getTransitionMatrix(self):
        '''
        Returns the transition matrix in algebraic form.

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [number of state x number of state]

        '''
        if self._transitionMatrixCompile is not None \
           or self._hasNewTransition == False:
            return self._transitionMatrix
        else:
            return super(SimulateOdeModel, self)._computeTransitionMatrix()
    
    def getTransitionVector(self):
        '''
        Returns the set of transitions in a single vector, transitions
        between state to state first then the birth and death process

        Returns
        -------
        :class:`sympy.matrices.matrices`
            A matrix of dimension [total number of transitions x 1]

        '''
        if self._transitionVectorCompile is not None or self._hasNewTransition:
            return self._transitionVector
        else:
            return super(SimulateOdeModel, self)._computeTransitionVector()

    def transitionMatrix(self, state, t):
        '''
        Evaluate the transition matrix given state and time

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
            a 2d array of size (M,M) where M is the number
            of transitions

        '''
        return self.evalTransitionMatrix(time=t, state=state)

    def evalTransitionMatrix(self, parameters=None, time=None, state=None):
        '''
        Evaluate the transition matrix given parameters, state and time. Note
        that the output is not in sparse format

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
        time: double
            The current time
        state: array list
            The current numerical value for the states which can
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of state x number of state]

        '''
        if self._transitionMatrixCompile is None or self._hasNewTransition:
            self._compileTransitionMatrix()

        evalParam = self._getEvalParam(state, time, parameters)
        return self._transitionMatrixCompile(evalParam)

    def _compileTransitionMatrix(self):
        '''
        We would also need to compile the function so that
        it can be evaluated faster.
        '''
        if self._transitionMatrix is None or self._hasNewTransition:
            super(SimulateOdeModel, self)._computeTransitionMatrix()

        f = self._SC.compileExprAndFormat
        if self._isDifficult:
            self._transitionMatrixCompile = f(self._sp,
                                              self._transitionMatrix,
                                              modules='mpmath')
        else:
            self._transitionMatrixCompile = f(self._sp,
                                              self._transitionMatrix)

        return None
    
    def transitionVector(self, state, t):
        '''
        Evaluate the transition vector given state and time

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
            a 1d array of size K where K is the number of between
            states transitions and the number of birth death
            processes
        '''
        return self.evalTransitionVector(time=t, state=state)

    def evalTransitionVector(self, parameters=None, time=None, state=None):
        '''
        Evaluate the transition vector given parameters, state and time. Note
        that the output is not in sparse format

        Parameters
        ----------
        parameters: list
            see :meth:`.setParameters`
        time: double
            The current time
        state: array list
            The current numerical value for the states which can
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            vector of dimension [total number of transitions]

        '''
        if self._transitionVectorCompile is None or self._hasNewTransition:
            self._compileTransitionVector()

        evalParam = self._getEvalParam(state, time, parameters)
        return self._transitionVectorCompile(evalParam)
    
    def _compileTransitionVector(self):
        '''
        We would also need to compile the function so that
        it can be evaluated faster.
        '''
        if self._transitionVector is None or self._hasNewTransition:
            super(SimulateOdeModel, self)._computeTransitionVector()

        f = self._SC.compileExprAndFormat
        if self._isDifficult:
            self._transitionVectorCompile = f(self._sp,
                                              self._transitionVector,
                                              modules='mpmath')
        else:
            self._transitionVectorCompile = f(self._sp,
                                              self._transitionVector)

        return None

    def getBirthDeathRate(self):
        '''
        Find the algebraic equations of birth and death processes

        Returns
        -------
        :class:`sympy.matrices.matrices`
            birth death process in matrix form
        '''
        if self._birthDeathRate is None or self._hasNewTransition:
            self._computeBirthDeathRate()

        return self._birthDeathRate

    def birthDeathRate(self, state, t):
        '''
        Evaluate the birth death rates given state and time

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
            an array of size (M,M) where M is the number
            of birth and death actions

        '''
        return self.evalBirthDeathRate(time=t, state=state)

    def evalBirthDeathRate(self, parameters=None, time=None, state=None):
        '''
        Evaluate the birth and death rates given parameters, state and time.

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
            Matrix of dimension [number of birth and death rates x 1]

        '''
        if self._birthDeathRateCompile is None or self._hasNewTransition:
            self._computeBirthDeathRate()

        evalParam = self._getEvalParam(state, time, parameters)
        return self._birthDeathRateCompile(evalParam)

    def _computeBirthDeathRate(self):
        '''
        Note that this is different to _birthDeathVector because
        this is of length (number of birth and death process) while
        _birthDeathVector in baseOdeModel has the same length as
        the number of states
        '''
        self._numBD = len(self._birthDeathList)
        # holder

        if self._numBD == 0:
            A = sympy.zeros(1, 1)
        else:
            A = sympy.zeros(self._numBD, 1)

            # go through all the transition objects
            for i, bd in enumerate(self._birthDeathList):
                A[i] += eval(self._checkEquation(bd.getEquation()))

        # assign back
        self._birthDeathRate = A
        # compilation of the symbolic calculation.  Note here that we are
        # going to recompile the total transitions because it might
        # have changed
        f = self._SC.compileExprAndFormat
        if self._isDifficult:
            self._birthDeathRateCompile = f(self._sp,
                                            self._birthDeathRate,
                                            modules='mpmath')
        else:
            self._birthDeathRateCompile = f(self._sp,
                                            self._birthDeathRate)

        return None

    def totalTransition(self, state, t):
        '''
        Evaluate the total transition rate given state and time

        Parameters
        ----------
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        float
            total rate

        '''
        return sum(self.transitionVector(time=t, state=state))

    def transitionMean(self, state, t):
        '''
        Evaluate the mean of the transitions given state and time.  For
        m transitions and n states, we have

        .. math::
            f_{j,k} &= \sum_{i=1}^{n} \\frac{\partial a_{j}(x)}{\partial x_{i}} v_{i,k} \\\\
            \\mu_{j} &= \sum_{k=1}^{m} f_{j,k}(x)a_{k}(x) \\\\
            \\sigma^{2}_{j}(x) &= \sum_{k=1}^{m} f_{j,k}^{2}(x) a_{k}(x)

        where :math:`v_{i,k}` is the state change matrix.

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
            an array of size m where m is the number of transition

        '''
        return self.evalTransitionMean(time=t, state=state)

    def evalTransitionMean(self, parameters=None, time=None, state=None):
        '''
        Evaluate the transition mean given parameters, state and time.

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

        '''
        if self._transitionMeanCompile is None or self._hasNewTransition:
            self._computeTransitionMeanVar()

        evalParam = self._getEvalParam(state, time, parameters)
        return(self._transitionMeanCompile(evalParam))

    def transitionVar(self, state, t):
        '''
        Evaluate the variance of the transitions given state and time

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
            an array of size M where M is the number of transition

        '''
        return(self.evalTransitionVar(time=t, state=state))

    def evalTransitionVar(self, parameters=None, time=None, state=None):
        '''
        Evaluate the transition variance given parameters, state and time.

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

        '''
        if self._transitionVarCompile is None or self._hasNewTransition:
            self._computeTransitionMeanVar()

        evalParam = self._getEvalParam(state, time, parameters)
        return(self._transitionVarCompile(evalParam))

    def _computeTransitionJacobian(self):
        if self._GMat is None:
            self._computeDependencyMatrix()

        F = sympy.zeros(self._numTransition, self._numTransition)
        for i in range(self._numTransition):
            for j, eqn in enumerate(self._transitionVector):
                for k, state in enumerate(self._iterStateList()):
                    diffEqn = sympy.diff(eqn, state, 1) 
                    tempEqn, isDifficult = simplifyEquation(diffEqn)
                    F[i,j] += tempEqn*self._vMat[k,i]
                    self._isDifficult = self._isDifficult or isDifficult
        
        self._transitionJacobian = F
        return(F)

    def _computeTransitionMeanVar(self):
        '''
        This is the mean and variance information that we need
        for the :math:`\tau`-Leap
        '''
    
        if self._transitionJacobian is None or self._hasNewTransition:
            self._computeTransitionJacobian()

        F = self._transitionJacobian
        # holders
        mu = sympy.zeros(self._numTransition, 1)
        sigma2 = sympy.zeros(self._numTransition, 1)
        # we calculate the mean and variance
        for i in range(self._numTransition):
            for j, eqn in enumerate(self._transitionVector):
                mu[i] += F[i,j] * eqn
                sigma2[i] += F[i,j] * F[i,j] * eqn

        self._transitionMean = mu
        self._transitionVar = sigma2

        # now we are going to compile them
        f = self._SC.compileExprAndFormat
        if self._isDifficult:
            self._transitionMeanCompile = f(self._sp,
                                            self._transitionMean,
                                            modules='mpmath')
            self._transitionVarCompile = f(self._sp,
                                           self._transitionVar,
                                           modules='mpmath')
        else:
            self._transitionMeanCompile = f(self._sp, self._transitionMean)
            self._transitionVarCompile = f(self._sp, self._transitionVar)

        return None

    ########################################################################
    #
    # Unrolling of ode to transitions
    #
    ########################################################################

    def returnObjWithTransitionsAndBD(self):
        '''
        Returns a :class:`SimulateOdeModel` with the same state and parameters
        as the current object but with the equations defined by a set of
        transitions and birth death process instead of say, odes
        '''
        transitionList = self.getTransitionsFromOde()
        bdList = self.getBDFromOde()

        return SimulateOdeModel(
                                [str(s) for s in self._stateList],
                                [str(p) for p in self._paramList],
                                derivedParamList=self._derivedParamEqn,
                                transitionList=transitionList,
                                birthDeathList=bdList
                                )

    def getTransitionsFromOde(self):
        '''
        Returns a list of :class:`Transition` from this object by unrolling
        the odes.  All the elements are of TransitionType.T
        '''
        M = self._generateTransitionMatrix()

        transitionList = list()
        for i, s1 in enumerate(self._stateList):
            for j, s2 in enumerate(self._stateList):
                if M[i,j] != 0:
                    t = Transition(origState=str(s1),
                                   destState=str(s2),
                                   equation=str(M[i,j]),
                                   transitionType=TransitionType.T)
                    transitionList.append(t)

        return(transitionList)
    
    def getBDFromOde(self, A=None):
        '''
        Returns a list of:class:`Transition` from this object by unrolling
        the odes.  All the elements are of TransitionType.B or
        TransitionType.D
        '''
        if A is None:
            if not ode_utils._noneOrEmptyList(self._odeList):
                eqnList = [t.getEquation() for t in self._odeList]
                A = sympy.Matrix(checkEquation(eqnList,
                                               *self._getListOfVariablesDict(),
                                               subsDerived=False))
            else:
                raise Exception("Object was not initialized using a set of ode")
            # A = super(SimulateOdeModel, self).getOde()

        bdList, _term = _ode_composition.getUnmatchedExpressionVector(A, True)

        if len(bdList) > 0:
            M = self._generateTransitionMatrix(A)

            A1 = _ode_composition.pureTransitionToOde(M)
            diffA = sympy.simplify(A - A1)

            # get our birth and death process
            bdUnroll = list()
            states = [str(i) for i in self.getStateList()]

            for i, a in enumerate(diffA):
                for b in bdList:
                    if _ode_composition._hasExpression(a, b):
                        if sympy.Integer(-1) in _ode_composition.getLeafs(b):
                            bdUnroll.append(Transition(origState=states[i],
                                            equation=str(b*-1),
                                            transitionType=TransitionType.D))
                        else:
                            bdUnroll.append(Transition(origState=states[i],
                                            equation=str(b),
                                            transitionType=TransitionType.B))
                        a -= b
            
            return bdUnroll
        else:
            return []

    def _generateTransitionMatrix(self, A=None):#, transitionExpressionList=None):
        '''
        Finds the transition matrix from the set of ode.  It is 
        important to note that although some of the functions used
        in this method appear to be the same as _getReactantMatrix
        and _getStateChangeMatrix, they are different in the sense
        that the functions called here is focused on the terms of 
        the equation rather than the states.
        '''
        if A is None:
            if not ode_utils._noneOrEmptyList(self._odeList):
                eqnList = [t.getEquation() for t in self._odeList]
                A = sympy.Matrix(checkEquation(eqnList,
                                               *self._getListOfVariablesDict(),
                                               subsDerived=False))
            else:
                raise Exception("Object was not initialized using a set of ode")

        bdList, _term = _ode_composition.getUnmatchedExpressionVector(A, True)
        fx = _ode_composition.stripBDFromOde(A, bdList)
        states = [s for s in self._iterStateList()]
        M, _remain = _ode_composition.odeToPureTransition(fx, states, True)
        return M
