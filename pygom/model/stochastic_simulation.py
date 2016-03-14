from _modelErrors import InputError, SimulationError
from pygom.utilR.distn import rexp, ppois, rpois, runif

import numpy

def directReaction(x, t, stateChangeMat, transitionFunc):
    '''
    The direct reaction method.  Same as :func:`firstReaction` for both
    input and output, only differ in internal computation
    '''
    
    rates = transitionFunc(x,t)
    totalRate = sum(rates)
    jumpRate = numpy.cumsum(rates)

    if totalRate > 0:
        jumpTime = rexp(1, totalRate)
        # U \sim \UnifDist[0,1]
        U = runif(1)
        targetRate = totalRate * U
        # find the index that covers the probability of jump using binary search
        transitionIndex = numpy.searchsorted(jumpRate, targetRate)
        # we can move!! move particles
        newX = _updateStateWithJump(x, transitionIndex, stateChangeMat)
        return _checkJump(x, newX, t, jumpTime)
    else:
        # we can't jump
        raise SimulationError("Cannot perform any more reactions")
        
def firstReaction(x, t, stateChangeMat, transitionFunc):
    '''
    The first reaction method
    
    Parameters
    ----------
    x: array like
        state vector
    t: double
        time
    stateChangeMat: array like
        State change matrix V_{i,j} where i = state and j = transition
        V_{i,j} is some non--zero integer such that transition j happen means
        that state i changes by V_{i,j} amount
    transitionFunc: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition rates

    Returns
    -------
    x: array like
        state vector
    t: double
        time
    success:
        if the leap was successful.  A change in both x and t if it is
        successful, no change otherwise
    '''
    rates = transitionFunc(x,t)
    # find our jump times
    jumpTimes = _newJumpTimes(rates)
    # first jump
    minIndex = numpy.argmin(jumpTimes)
    # validate the jump times
    if jumpTimes[minIndex] == numpy.Inf:
        # if we cannot perform any more jumps
        raise SimulationError("Cannot perform any more reactions")
    else:
        newX = _updateStateWithJump(x, minIndex, stateChangeMat)
        return _checkJump(x, newX, t, jumpTimes[minIndex])

def nextReaction(x, t, stateChangeMat, dependencyGraph, oldRates, jumpTimes, transitionFunc):
    '''
    The next reaction method
    '''

    # smallest time :)
    index = numpy.argmin(jumpTimes)
    # moving state and time
    newX = _updateStateWithJump(x, index, stateChangeMat)
    t = jumpTimes[index]
    # recalculate the new transition matrix
    if hasattr(transitionFunc, '__call__'):
        rates = transitionFunc(x,t)
        # update the jump time
        jumpTimes[index] = t + rexp(1, rates[index])
    elif hasattr(transitionFunc, '__iter__'):
        jumpTimes[index] = t + rexp(1, transitionFunc[index](x, t))
    else:
        raise InputError("transitionFunc should be a callable or a list of callable")

    # then go through the remaining transitions
    for i, anew in enumerate(rates):
        # obviously, not the target transition as we have already fixed it
        if i != index:
            # and only if the rate has been affected by the state update
            if dependencyGraph[i,index] != 0:
                aold = oldRates[i]
                if anew > 0:
                    jumpTimes[i] = (aold/anew)*(jumpTimes[i] - t) + t
                else:
                    jumpTimes[i] = numpy.Inf
        # done :)
        return newX, t, True, rates, jumpTimes
    else:
        raise SimulationError("Cannot perform any more reactions")


def tauLeap(x, t, stateChangeMat, reactantMat, transitionFunc, transitionMeanFunc, transitionVarFunc, epsilon=0.1):
    '''
    The Poisson :math:`\tau`-Leap
    
    Parameters
    ----------
    x: array like
        state vector
    t: double
        time
    stateChangeMat: array like
        State change matrix V_{i,j} where i = state and j = transition
        V_{i,j} is some non--zero integer such that transition j happen means
        that state i changes by V_{i,j} amount
    reactantMat:array like
        Reactant matrix of \lambda_{i,j} where i = state and j = transition
        A value of 1 if state i is involved in transition j
    transitionFunc: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition rates
    transitionMean: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition mean
    transitionVar: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition variance
    epsilon: double, optional
        tolerence of the size of the jump, defaults to 0.1
    
    Returns
    -------
    x: array like
        state vector
    t: double
        time
    success:
        if the leap was successful.  A change in both x and t if it is
        successful, no change otherwise
    '''
    # go through the list of transitions
    rates = transitionFunc(x,t)
    totalRate = sum(rates)

    exceedCDFArray = list()
    safeToJump = False

    mu = transitionMeanFunc(x, t)
    sigma2 = transitionVarFunc(x, t)
    # then we go find out the condition
    # \min_{j \in \left[1,M\right]} \{ l,r \}
    # where l = \gamma / \abs(\mu_{j}(x)) ,
    # and r = \gamma^{2} / \sigma_{j}^{2}(x)
    top = epsilon * totalRate
    l = top / abs(mu)
    r = top**2 / sigma2
    tauScale = min(min(l), min(r))
    # note that the above calculation is actually very slow, because
    # we can rewrite the conditions into
    # \min \{ \min_{j \in \left[1,M\right]} l , \min_{j \in \left[1,M\right]} r \}
    # which again can be further simplified into
    # \gamma / \max_{j \in \left[1,\M\right]} \{ \abs(\mu_{j}(x),\sigma_{j}^{2} \}

    # we put in an additional safety mechanism here where we also evaluate
    # the probability that a realization exceeds the observations and further
    # decrease the time step.
    while safeToJump == False:
        for i, r in enumerate(rates):
            activeX = x[reactantMat[:,i]]
            for xi in activeX:
                exceedCDFArray.append(ppois(xi, tauScale*r))

        # the expected probability that our jump will exceed the value
        maxExceed = numpy.max(1.0 - numpy.array(exceedCDFArray))
        # cannot allow it to exceed out epsilon
        if maxExceed > epsilon:
            tauScale /= 2.0
        else:
            safeToJump = True
    ## end while safeToJump==False

    # print tauScale
    # print rates

    # make the jumps
    newX = x.copy()
    for i, r in enumerate(rates):
        # realization
        jumpQuantity = rpois(1, tauScale*r)
        # print jumpQuantity
        # move the particles!
        newX = _updateStateWithJump(newX, i, stateChangeMat, jumpQuantity)
        ## done moving
    return _checkJump(x, newX, t, tauScale)

def _newJumpTimes(rates):
    '''
    Generate the new jump times assuming that the rates follow an exponential
    distribution
    '''   
    tau = [rexp(1, r) if r > 0 else numpy.Inf for r in rates]
    return numpy.array(tau)

def _updateStateWithJump(x, transitionIndex, stateChangeMat, n=1.0):
    '''
    Updates the states given a jump.  Makes use the state change
    matrix, and updates according to the number of times this
    transition has happened
    '''
    return x + stateChangeMat[:,transitionIndex] * n
    
def _checkJump(x, newX, t, jumpTime):
    failedJump = numpy.any(newX < 0)

    if failedJump:
        return x, t, False
    else:
        t += jumpTime
        return newX, t, True
