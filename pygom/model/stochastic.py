"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module/class that carries out different type of simulation
    on an ode formulation

"""

__all__ = ['SimulateOdeModel']

from .deterministic import OperateOdeModel
from .transition import TransitionType, Transition
from _modelErrors import InputError, SimulationError
from pygom.utilR.distn import rexp, runif, rpois, ppois
import ode_utils
import ode_composition

import numpy
import sympy
import scipy.stats
import copy

modulesE = ['numpy', 'mpmath', 'sympy']
modulesH = ['mpmath', 'sympy']

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

        # just to confirm, we DO NEED the transition matrix here
        self._transitionMatrix = None
        self._transitionMatrixCompile = None

        self._birthDeathRate = None
        self._birthDeathRateCompile = None

        # information required for the tau leap
        self._transitionJacobian = None
        self._totalTransition = None

        self._transitionMean = None
        self._transitionMeanCompile = None

        self._transitionVar = None
        self._transitionVarCompile = None

        self._totalTransitionCompile = None

        # information for the next reaction method
        self._lambdaMat = None
        self._vMat = None
        self._GMat = None
        # micro times for jumps
        self._tau = None
        self._tauDict = None

        self._EPSILON = 0.01
        self._tauScale = 0.1

        self._numT = len(self._transitionList)
        self._numBD = len(self._birthDeathList)

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
        full_output: bool
            if we want additional information

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

            dview, canParallel = self._setupParallel(t, iteration, self._stochasticParam)
            if canParallel:
                print "Parallel"
                dview.execute('solutionList = [odeS.integrate(t) for i in range(iteration)]')
                solutionList = list()
                for i in dview['solutionList']:
                    solutionList += i
            else:
                raise Exception("Cannot run this in parallel")

        except Exception as e:
            print e
            print "Serial"
            solutionList = [self.integrate(t) for i in range(iteration)]

        # now make our 3D array
        # the first dimension is the number of iteration
        Y = numpy.array(solutionList)

        if full_output:
            return numpy.mean(Y, axis=0), solutionList
        else:
            return numpy.mean(Y, axis=0)

    def simulateJump(self, t, iteration, full_output=False):
        '''
        Simulate the ode using stochastic simulation.  It switches
        between a first reaction method and a :math:`\tau`-leap
        algorithm internally.

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
            or the final time point
        iteration: int
            number of iterations you wish to simulate
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

        assert len(self._odeList) == 0, "Currently only able to simulate when only transitions are present"
        assert not numpy.all(self._x0<=1.0), "Can only simulate a jump process with non-normalized populations"
#         if len(self._odeList) != 0:
#             raise InputError("Currently only able to simulate when only transitions are present")
        #if len(self._birthDeathList)!=0:
        #    raise Exception("Currently only able to simulate when only transitions are present")

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

        if self._transitionMatrix is None:
            self._computeTransitionMatrix()

        # we are going to try the parallel option
        try:
            # check the type of parameter we have as input
            dview, canParallel = self._setupParallel(finalT, iteration, self._paramValue)
            if canParallel:
                #print "Parallel"
                dview.execute('YList = [odeS._jump(t,full_output=True) for i in range(iteration)]')
                # unroll information
                simXList = list()
                simTList = list()
                for Y in dview['YList']:
                    for simOut in Y:
                        simXList.append(simOut[0])
                        simTList.append(simOut[1])
                #print "Finished"
            else:
                raise SimulationError("Cannot run this in parallel")

        except Exception as e:
            #print "Serial"
            simXList = list()
            simTList = list()
            for i in range(iteration):
                # now we simulate the jumps
                simX, simT = self._jump(finalT, full_output=True)
                # add to list :)
                simXList.append(simX)
                simTList.append(simT)

        # now we want to fix our simulation, if they need fixing that is
        # print timePoint
        if timePoint:
            for i in range(len(simXList)):
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
        observe, we extract the observations x_{t} with
        \min\{ \abs( t - targetTime) \}
        '''
        y = list()
        maxTime = max(t)
        index = 0
        for i in targetTime:
            # we do not go and find the new time because
            # the simulation has reached saturation
            if i > maxTime:
                y.append(X[index,:])
            else:
                index = numpy.searchsorted(t, i)
                y.append(X[index,:])

        return numpy.array(y)

    def _jump(self, finalT, full_output=True):
        '''
        Jumps from the initial time self._t0 to the input time finalT
        '''
        # initial time
        assert self._t0 is not None, "No initial time"
        assert self._x0 is not None, "No initial state"
#         if self._t0 is None:
#             raise InitializeException("No initial time")
#         if self._x0 is None:
#             raise InitializeException("No initial state")
        t = self._t0.tolist()
        x = copy.deepcopy(self._x0)

        # holders
        xList = list()
        tList = list()

        # record information
        xList.append(x.copy())
        tList.append(t)

        # we want to construct some jump times
        A = self.transitionMatrix(self._x0, self._t0)
        BD = self.birthDeathRate(x, t)

        self._newJumpTimes(A, BD)

        if sum(self._tauDict.values()) == 0:
            raise SimulationError("Initial Conditions does not produce any jumps")

        # keep jumping, Whoop Whoop (put your hands up!).
        while t < finalT:
            # print x
            # obtain our transition matrix
            A = self.transitionMatrix(x, t)
            BD = self.birthDeathRate(x, t)
            try:
                if numpy.min(x) > 10:
                    # print "\nLeap"
                    # print "state" +str(x)
                    # first we will try the \tau-leap.  If it fails, for whatever
                    # reason, then we will move on the the first reaction method.
                    x, t, success = self._tauLeap(x, t, A, BD)
                    # x, t, success = self._firstReaction(x, t, A, BD)
                    # print x
                    # print success
                    if success is False:
                        x, t, success = self._firstReaction(x, t, A, BD)
                else:
                    # print "\nFirst"
                    # print "state" +str(x)
                    x, t, success = self._firstReaction(x, t, A, BD)
                # x,t,success = self._directReaction(x, t, A, numTransition)
                if success:
                    xList.append(x.copy())
                    tList.append(t)
                # else:
                #     pass # break
            except SimulationError:
                break
        return numpy.array(xList), numpy.array(tList)

    def _tauLeap(self, x, t, A, BD):
        '''
        The Poisson :math:`\tau`-Leap
        '''
        # go through the list of transitions
        numTransition = self._numT + self._numBD
        rateArray = numpy.zeros(numTransition)
        exceedCDFArray = numpy.zeros(numTransition)
        safeToJump = False

        # to compute the tau leap time point, condition
        totalRate = self.totalTransition(x, t)
        # magic!
        mu = self.transitionMean(x, t)
        sigma2 = self.transitionVar(x, t)
        # then we go find out the condition
        # \min_{j \in \left[1,M\right]} \{ l,r \}
        # where l = \gamma / \abs(\mu_{j}(x)) ,
        # and r = \gamma^{2} / \sigma_{j}^{2}(x)
        top = self._EPSILON * totalRate
        l = top / abs(mu)
        r = top**2 / sigma2
        self._tauScale = min(min(l), min(r))
        # note that the above calculation is actually very slow, because
        # we can rewrite the conditions into
        # \min \{ \min_{j \in \left[1,M\right]} l , \min_{j \in \left[1,M\right]} r \}
        # which again can be further simplified into
        # \gamma / \max_{j \in \left[1,\M\right]} \{ \abs(\mu_{j}(x),\sigma_{j}^{2} \}

        # we put in an additional safety mechanism here where we also evaluate
        # the probability that a realization exceeds the observations and further
        # decrease the time step.
        while safeToJump == False:
            totalRate = 0
            for i in range(0, numTransition):
                # cannot use function call here because we need
                # the fromIndex later
                if i < self._numT:
                    transObj = self._transitionList[i]
                    fromIndex = self._extractStateIndex(transObj.getOrigState())
                    toIndex = self._extractStateIndex(transObj.getDestState())
                    rateArray[i] = A[fromIndex,toIndex]
                else:
                    bdObj = self._birthDeathList[i-self._numT]
                    fromIndex = self._extractStateIndex(bdObj.getOrigState())
                    rateArray[i] = BD[fromIndex]
                    # how much probability we are covering
                exceedCDFArray = ppois(x[fromIndex], self._tauScale*rateArray[i])
            ## end of loop

            # the expected probability that our jump will exceed the value
            maxExceed = numpy.max(1-exceedCDFArray)
            # cannot allow it to exceed out epsilon
            if maxExceed > 0.001:
                self._tauScale /= 2.0
            else:
                safeToJump = True
        ## end while safeToJump==False

        # make the jumps
        newX = x.copy()
        for i in range(0, numTransition):
            # realization
            jumpQuantity = rpois(1, self._tauScale*rateArray[i])
            # move the particles!
            newX = self._updateStateWithJump(newX, i, jumpQuantity)
            ## done moving
        return self._checkJump(x, newX, t, self._tauScale)

    def _firstReaction(self, x, t, A, BD):
        '''
        The first reaction method
        '''
        # find our jump times
        jumpTime = self._newJumpTimes(A, BD)

        # first jump
        minIndex = numpy.argmin(numpy.array(jumpTime))
        # validate the jump times
        if jumpTime[minIndex] == numpy.Inf:
            # if we cannot perform any more jumps
            raise SimulationError("Cannot perform any more reactions")
        else:
            newX = self._updateStateWithJump(x, minIndex)
            return self._checkJump(x, newX, t, jumpTime[minIndex])

    # @deprecated
    def _directReaction(self, x, t, A, BD):
        '''
        The direct reaction method
        '''
        jumpRate = numpy.ones(self._numT + self._numBD)
        totalRate = 0.0
        for i in range(0, self._numT):
            rate = self._getRateWithIndex(A, i)
            # note that \lambda = 1/rate for the exponential
            # we want to ensure that the rate is feasible
            totalRate += rate
            jumpRate[i] = totalRate

        for i in range(0, self._numBD):
            j = i + self._numT
            rate = BD[i]
            totalRate += rate
            jumpRate[j] = totalRate

        if totalRate > 0:
            jumpTime = rexp(totalRate)
            # U \sim \UnifDist[0,1]
            U = runif(1)
            targetRate = totalRate * U
            # find the index that covers the probability of jump using binary search
            transitionIndex = numpy.searchsorted(numpy.array(jumpRate), targetRate)
            # we can move!! move particles
            newX = self._updateStateWithJump(x, transitionIndex)
            return self._checkJump(x, newX, t, jumpTime)
        else:
            # we can't jump
            raise SimulationError("Cannot perform any more reactions")
    # @deprecated
    def _nextReaction(self, x, t, oldA, numTransition):
        '''
        The next reaction method
        '''
        # get the required information
        if self._GMat is None:
            self._getDependencyMatrix()

        # smallest time :)
        index = min(self._tauDict, key=self._tauDict.get)
        # moving state and time
        x = self._updateStateWithJump(x, index)
        t = self._tauDict[index]
        # recalculate the new transition matrix
        A = self.transitionMatrix(x, t)

        # only proceed if we have feasible transitions
        if A.sum() > 0:
            # we can jump
            # first, find out the new jump time for the moved transition
            #self._tauDict[index] = t + self._rexp.rvs(1)[0]/self._getRateWithIndex(A,index)
            self._tauDict[index] = t + rexp(self._getRateWithIndex(A, index))
            # then go through the remaining transitions
            for i in range(0, numTransition):
                # obviously, not the target transition as we have already fixed it
                if i != index:
                    # and only if the rate has been affected by the state update
                    if self._GMat[i,index] != 0:
                        aold = self._getRateWithIndex(oldA, i)
                        anew = self._getRateWithIndex(A, i)
                        if anew > 0:
                            self._tauDict[i] = (aold/anew) * (self._tauDict[i] - t) + t
                        else:
                            self._tauDict[i] = numpy.Inf

            # done :)
            return x, t, True
        else:
            raise SimulationError("Cannot perform any more reactions")
            # return x, t, False

    ########################################################################
    #
    # Jumping mechanism
    #
    ########################################################################

    def _getRateWithIndex(self, A, index):
        '''
        Obtain the rate from the matrix A using the index of the ode.
        '''
        transObj = self._transitionList[index]
        fromIndex = self._extractStateIndex(transObj.getOrigState())
        toIndex = self._extractStateIndex(transObj.getDestState())
        return A[fromIndex, toIndex]

    def _newJumpTimes(self, A, BD):
        '''
        Generate a new set of jump times for all the transitions
        '''
        # redefine
        self._tau = numpy.ones(self._numT+self._numBD) * numpy.Inf
        self._tauDict = dict()
        for i in range(0, self._numT):
            rate = self._getRateWithIndex(A, i)
            # note that \lambda = 1/rate for the exponential
            # we want to ensure that the rate is feasible
            if rate > 0:
                self._tau[i] = rexp(1, rate)

            # bind regardless. This is actually a massive waste of
            # memory.  As much as self._numT * size of a double aka
            # 64 bits!  Yes, old machine with 1mb RAM, you becareful
            # there.
            self._tauDict[i] = self._tau[i]

        for i in range(0, self._numBD):
            j = i + self._numT
            rate = BD[i]
            # check the validity of the rate
            if rate > 0:
                self._tau[j] = rexp(1, rate)

            # bind regardless... even if we have
            # no updated time!?
            self._tauDict[j] = self._tau[j]

        return self._tau

    def _updateStateWithJump(self, x, index, n=1):
        '''
        Updates the states given a jump.  The default is that
        our jumps are of single particle, but multiple are
        also allowed
        '''
        newX = x.copy()
        if index < self._numT:
            #print "Normal move = "+str(n)
            transObj = self._transitionList[index]
            fromIndex = self._extractStateIndex(transObj.getOrigState())

            toIndex = self._extractStateIndex(transObj.getDestState())
            #print "from : " +str(fromIndex)+ " to : " +str(toIndex)
            # move a particle
            newX[fromIndex] -= n
            newX[toIndex] += n
        else:
            #print "BD move with n = "+str(n)
            index -= self._numT
            BDObj = self._birthDeathList[index]

            fromIndex = self._extractStateIndex(BDObj.getOrigState())
            # print "index = " +str(fromIndex)+  " with type " +str(BDObj.getTransitionType())
            # check which type of move it is
            if BDObj.getTransitionType() is TransitionType.B:
                # We are giving births!
                newX[fromIndex] += n
            elif BDObj.getTransitionType() is TransitionType.D:
                # The system is dying :(
                newX[fromIndex] -= n

        return newX
    
    def _checkJump(self, x, newX, t, jumpTime):
        failedJump = numpy.any(newX < 0)

        if failedJump:
            return x, t, False
        else:
            t += jumpTime
            return newX, t, True

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
        if self._transitionMatrixCompile is not None:
            return self._transitionMatrix
        else:
            self._computeTransitionMatrix()
            return self._transitionMatrix

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

        if state is None or time is None:
            raise InputError("Have to input both state and time")

        if self._transitionMatrixCompile is None:
            self._computeTransitionMatrix()

        # we should usually only want the time and state
        # and the parameters will be fixed
        if parameters is not None:
            self.setParameters(parameters)
        elif self._parameters is None:
            if len(self._paramList) != 0:
                raise InputError("Have not set the parameters yet")

        # output, substitute in the numerics if they are available
        if type(state) is list:
            evalParam = state + [time]
        else:
            evalParam = list(state) + [time]

        evalParam += self._paramValue
        return self._transitionMatrixCompile(evalParam)

    def _computeTransitionMatrix(self):
        '''
        We would also need to compile the function so that
        it can be evaluated faster.
        This is an override - Majority of the code here is
        identical to the one originally in BaseOdeModel.
        Difference being we have to take into account the
        total transition which also includes the birth
        death vector.  We also compile the code here.

        '''

        # holders
        self._transitionMatrix = sympy.zeros(self._numState, self._numState)
        self._totalTransition = sympy.zeros(1, 1)

        A = self._transitionMatrix
        # going through the list of transition
        for i in range(0, len(self._transitionList)):
            # extract the object
            # could have equally done this via the line
            # for transition in self._transitionList:
            # but I prefer to loop using range. Bite me.
            transition = self._transitionList[i]
            # then find out the indices of the states
            fromIndex = self._extractStateIndex(transition.getOrigState())
            toIndex = self._extractStateIndex(transition.getDestState())

            # put the getEquation in the correct element
            tempTransition = eval(self._checkEquation(transition.getEquation()))
            A[fromIndex,toIndex] += tempTransition
            # simplify if possible
            A[fromIndex,toIndex] = super(SimulateOdeModel,
                                         self)._simplifyEquation(A[fromIndex,toIndex])
            # we also want the total
            self._totalTransition[0,0] += tempTransition
        # assign back
        self._transitionMatrix = A

        if self._isDifficult:
            self._transitionMatrixCompile = self._SC.compileExprAndFormat(self._sp,
                                                                          self._transitionMatrix,
                                                                          modules=modulesH)
            self._totalTransitionCompile = self._SC.compileExprAndFormat(self._sp,
                                                                         self._totalTransition,
                                                                         modules=modulesH)
        else:
            self._transitionMatrixCompile = self._SC.compileExprAndFormat(self._sp,
                                                                          self._transitionMatrix)
            self._totalTransitionCompile = self._SC.compileExprAndFormat(self._sp,
                                                                         self._totalTransition)

        return None

    def getBirthDeathRate(self):
        '''
        Find the algebraic equations of birth and death processes

        Returns
        -------
        :class:`sympy.matrices.matrices`
            birth death process in matrix form
        '''
        if self._birthDeathRate is None:
            self._computeBirthDeathRate()
        else:
            pass

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

        if state is None or time is None:
            raise InputError("Have to input both state and time")

        if self._birthDeathRateCompile is None:
            self._computeBirthDeathRate()

        # we should usually only want the time and state
        # and the parameters will be fixed
        if parameters is not None:
            self.setParameters(parameters)
        elif self._parameters is None:
            if len(self._paramList) != 0:
                raise InputError("Have not set the parameters yet")

        # output, substitute in the numerics if they are available
        if type(state) is list:
            evalParam = state + [time]
        else:
            evalParam = list(state) + [time]

        evalParam += self._paramValue
        return self._birthDeathRateCompile(evalParam)

    def _computeBirthDeathRate(self):
        '''
        Note that this is different to _birthDeathVector because
        this is of length (number of birth and death process) while
        _birthDeathVector in baseOdeModel has the same length as
        the number of states
        '''
        numBD = len(self._birthDeathList)
        # holder

        if numBD == 0:
            A = sympy.zeros(1, 1)
        else:
            A = sympy.zeros(numBD, 1)

            # go through all the transition objects
            for i in range(0, len(self._birthDeathList)):
                # extract object
                bd = self._birthDeathList[i]
                # everything is of a positive sign because they
                # are rates
                A[i] += eval(self._checkEquation(bd.getEquation()))

                # everything is part of total transition!
                self._totalTransition += A[i]

        # assign back
        self._birthDeathRate = A
        # compilation of the symbolic calculation.  Note here that we are going to
        # recompile the total transitions because it might have changed
        if self._isDifficult:
            self._birthDeathRateCompile = self._SC.compileExprAndFormat(self._sp,
                                                                        self._birthDeathRate,
                                                                        modules=modulesH)
            self._totalTransitionCompile = self._SC.compileExprAndFormat(self._sp,
                                                                         self._totalTransition,
                                                                         modules=modulesH)
        else:
            self._birthDeathRateCompile = self._SC.compileExprAndFormat(self._sp,
                                                                        self._birthDeathRate)
            self._totalTransitionCompile = self._SC.compileExprAndFormat(self._sp,
                                                                         self._totalTransition)

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
        return self.evalTotalTransition(time=t, state=state)

    def evalTotalTransition(self, parameters=None, time=None, state=None):
        '''
        Evaluate the total transition given parameters,time and state

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

        if state is None or time is None:
            raise InputError("Have to input both state and time")

        if self._totalTransitionCompile is None or self._totalTransition is None:
            self._computeTransitionMatrix()

        # we should usually only want the time and state
        # and the parameters will be fixed
        if parameters is not None:
            self.setParameters(parameters)
        elif self._parameters is None:
            raise InputError("Have not set the parameters yet")

        # output, substitute in the numerics if they are available
        if isinstance(state, list):
            evalParam = state + [time]
        else:
            evalParam = list(state) + [time]

        evalParam += self._paramValue
        return self._totalTransitionCompile(evalParam)

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

        if state is None or time is None:
            raise InputError("Have to input both state and time")

        if self._transitionMeanCompile is None:
            self._computeTransitionMeanVar()

        # we should usually only want the time and state
        # and the parameters will be fixed
        if parameters is not None:
            self.setParameters(parameters)
        elif self._parameters is None:
            raise InputError("Have not set the parameters yet")

        # output, substitute in the numerics if they are available
        if isinstance(state, list):
            evalParam = state + [time]
        else:
            evalParam = list(state) + [time]

        evalParam += self._paramValue
        return self._transitionMeanCompile(evalParam)

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
        return self.evalTransitionVar(time=t, state=state)

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

        if state is None or time is None:
            raise InputError("Have to input both state and time")

        if self._transitionVarCompile is None:
            self._computeTransitionMeanVar()

        # we should usually only want the time and state
        # and the parameters will be fixed
        if parameters is not None:
            self.setParameters(parameters)
        elif self._parameters is None:
            raise InputError("Have not set the parameters yet")

        # output, substitute in the numerics if they are available
        if type(state) is list:
            evalParam = state + [time]
        else:
            evalParam = list(state) + [time]

        evalParam += self._paramValue
        return self._transitionVarCompile(evalParam)

    def _computeTransitionJacobian(self):
        numTransitions = self._numT + self._numBD
        if self._GMat is None:
            self._getDependencyMatrix()

        # holders
        A = self.getTransitionMatrix()
        BD = self.getBirthDeathRate()

        F = sympy.zeros(numTransitions, numTransitions)
        # going through the list of transition
        for k in range(0, numTransitions):
            for j in range(0, numTransitions):
                # although the method before is called getRateWithIndex,
                # it in fact only extract the element without regard to
                # the type of matrix it is
                if j < self._numT:
                    eqn = self._getRateWithIndex(A, j)
                else:
                    eqn = BD[j-self._numT]
                # assign
                for i in range(0, self._numState):
                    diffEqn = super(SimulateOdeModel, self)._simplifyEquation(sympy.diff(eqn, self._stateList[i], 1) ) * self._vMat[i,k]
                    F[k,j] += diffEqn

        # now we assign... no idea why this should b e done separately
        self._transitionJacobian = F

        return None

    def _computeTransitionMeanVar(self):
        '''
        This is the mean and variance information that we need
        for the :math:`\tau`-Leap
        '''
        numTransitions = self._numT + self._numBD

        if self._transitionJacobian is None:
            self._computeTransitionJacobian()

        F = self._transitionJacobian

        # holders
        mu = sympy.zeros(numTransitions, 1)
        sigma2 = sympy.zeros(numTransitions, 1)
        A = self._transitionMatrix
        BD = self._birthDeathRate
        # we calculate the mean and variance
        for i in range(0, numTransitions):
            for j in range(0, numTransitions):
                # find the correct position
                if j < self._numT:
                    eqn = self._getRateWithIndex(A,j)
                else:
                    eqn = BD[j-self._numT]
                # now the mean and variance
                mu[i] += F[i,j] * eqn
                sigma2[i] += F[i,j] * F[i,j] * eqn

        self._transitionMean = mu
        self._transitionVar = sigma2

        # now we are going to compile them
        if self._isDifficult:
            self._transitionMeanCompile = self._SC.compileExprAndFormat(self._sp,
                                                                        self._transitionMean,
                                                                        modules=modulesH)
            self._transitionVarCompile = self._SC.compileExprAndFormat(self._sp,
                                                                       self._transitionVar,
                                                                       modules=modulesH)

        else:
            self._transitionMeanCompile = self._SC.compileExprAndFormat(self._sp, self._transitionMean)
            self._transitionVarCompile = self._SC.compileExprAndFormat(self._sp, self._transitionVar)

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
        return SimulateOdeModel([str(s) for s in self._stateList],
                                [str(p) for p in self._paramList],
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
                if M[i,j]!=0:
                    t = Transition(origState=str(s1),
                                   destState=str(s2),
                                   equation=str(M[i,j]),
                                   transitionType=TransitionType.T)
                    transitionList.append(t)

        return transitionList
    
    def getBDFromOde(self, A=None, transitionExpressionList=None):
        '''
        Returns a list of:class:`Transition` from this object by unrolling
        the odes.  All the elements are of TransitionType.B or
        TransitionType.D
        '''
        if A is None:
            A = super(SimulateOdeModel, self).getOde()

        if transitionExpressionList is None:
            transitionList = ode_composition.getMatchingExpressionVector(A, True)
        else:
            transitionList = transitionExpressionList

        bdList = ode_composition.getUnmatchedExpressionVector(A, False)
        if len(bdList) > 0:
            M = self._generateTransitionMatrix(A, transitionList)

            # reduce the original set of ode to only birth and death process remaining
            M1 = M - M.transpose()
            diffA = sympy.zeros(self._numState,1)
            for i in range(self._numState):
                a = sympy.simplify(sum(M1[i,:]))
                diffA[i] = sympy.simplify(a + A[i])

            # get our birth and death process
            bdListUnroll = list()
            states = [str(i) for i in self.getStateList()]

            for i, a in enumerate(diffA):
                for b in bdList:
                    if ode_composition._hasExpression(a, b):
                        if sympy.Integer(-1) in ode_composition.getLeafs(b):
                            bdListUnroll.append(Transition(origState=states[i],
                                                equation=str(b*-1),
                                                transitionType=TransitionType.D))
                        else:
                            bdListUnroll.append(Transition(origState=states[i],
                                                equation=str(b),
                                                transitionType=TransitionType.B))
                    a -= b
            
            return bdListUnroll
        else:
            return []

    def _generateTransitionMatrix(self, A=None, transitionExpressionList=None):
        '''
        Finds the transition matrix from the set of ode.  It is 
        important to note that although some of the functions used
        in this method appear to be the same as _getReactantMatrix
        and _getStateChangeMatrix, they are different in the sense
        that the functions called here is focused on the terms of 
        the equation rather than the states.
        '''
        if A is None:
            A = super(SimulateOdeModel, self).getOde()

        if transitionExpressionList is None:
            transitionList = ode_composition.getMatchingExpressionVector(A, True)
        else:
            transitionList = transitionExpressionList
            
        B = ode_composition.generateDirectedDependencyGraph(A, transitionList)
        numRow, numCol = B.shape

        M = sympy.zeros(numRow)
        for j in range(numCol):
            i = B[:,j].argmax()
            k = B[:,j].argmin()
            M[i,k] += transitionList[j][0]

        return M

    ########################################################################
    #
    # Other matrix related to stochastic simulation
    #
    ########################################################################

    def _getReactantMatrix(self):
        '''
        The reactant matrix, where

        .. math::
            \lambda_{i,j} = \left\{ 1, &if state i is involved in transition j, \\
                                    0, &otherwise \right.
        '''
        numTransition = self._numT + self._numBD

        # declare holder
        self._lambdaMat = numpy.zeros((self._numState, numTransition))

        for j in range(0, numTransition):
            if j < self._numT:
                transObj = self._transitionList[j]
                # then find out the indices of the states
                fromIndex = self._extractStateIndex(transObj.getOrigState())
                toIndex = self._extractStateIndex(transObj.getDestState())
                # get the actual getEquation out
                eqn = self._transitionMatrix[fromIndex, toIndex]
            else:
                bdObj = self._birthDeathList[j-self._numT]
                fromIndex = self._extractStateIndex(bdObj.getOrigState())
                eqn = self._birthDeathRate[fromIndex]

            # now go through all the states
            for i in range(0, self._numState):
                if self._stateList[i] in eqn.atoms():
                    self._lambdaMat[i,j] = True

        return None

    def _getStateChangeMatrix(self):
        '''
        The state change matrix, where
        .. math::
            v_{i,j} = \left\{ 1, &if transition j cause state i to lose a particle, \\
                             -1, &if transition j cause state i to gain a particle, \\
                              0, &otherwise \right.
        '''
        numTransition = self._numT + self._numBD

        # declare holder
        self._vMat = numpy.zeros((self._numState, numTransition))

        for j in range(0, numTransition):
            if j < self._numT:
                transObj = self._transitionList[j]
                # then find out the indices of the states
                fromIndex = self._extractStateIndex(transObj.getOrigState())
                toIndex = self._extractStateIndex(transObj.getDestState())
                self._vMat[fromIndex,j] = -1
                self._vMat[toIndex,j] = 1
            else:
                bdObj = self._birthDeathList[j - self._numT]
                # then find out the indices of the states
                fromIndex = self._extractStateIndex(bdObj.getOrigState())
                if bdObj.getTransitionType() is TransitionType.B:
                    self._vMat[fromIndex,j] = 1
                elif bdObj.getTransitionType() is TransitionType.D:
                    self._vMat[fromIndex,j] = -1

        return None

    def _getDependencyMatrix(self):
        '''
        Obtain the dependency matrix/graph
        '''
        if self._lambdaMat is None:
            self._getReactantMatrix()
        if self._vMat is None:
            self._getStateChangeMatrix()

        numTransition = self._numT + self._numBD
        self._GMat = numpy.zeros((numTransition, numTransition))

        for i in range(0, numTransition):
            for j in range(0, numTransition):
                d = 0
                for k in range(0,self._numState):
                    d = d or (self._lambdaMat[k,i] and self._vMat[k,i])
                self._GMat[i,j] = d

        return None


    ########################################################################
    #
    # Setting up the parallel environment
    #
    ########################################################################

    def _setupParallel(self, t, iteration, paramEval):
        '''
        Try and setup an environment for parallel computing

        Parameters
        ----------
        paramEval: object
            This can be a dictionary, tuple, list, whatever type that we take in as parameters

        '''
        try:
            from IPython.parallel import Client
            rc = Client(profile='mpi')
            dview = rc[:]
            numCore = len(rc.ids)
            #print "The number of cores = " +str(numCore)

            dview.block = True
            # information required to setup the ode object
            dview.push(dict(stateList=self._stateList,
                            paramList=self._paramList,
                            derivedParamList=self._derivedParamList,
                            transitionList=self._transitionList,
                            birthDeathList=self._birthDeathList,
                            odeList=self._odeList))

            # initial conditions
            dview.push(dict(x0=self._x0, t0=self._t0, t=t, paramEval=paramEval))
            # and the number of iteration, we always run more or equal to the
            # number of iterations desired
            dview.push(dict(iteration=iteration/numCore + 1))

            # now run the commands that will initialize the models
            dview.execute('from pygom import SimulateOdeModel', block=True)
            dview.execute('odeS = SimulateOdeModel([str(i) for i in stateList],[str(i) for i in paramList],derivedParamList,transitionList,birthDeathList,odeList)', block=True)
            dview.execute('odeS.setInitialValue(x0,t0).setParameters(paramEval)', block=True)
            return dview, True
        except Exception as e:
            print e
            return e, False
