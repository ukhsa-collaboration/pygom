from _modelErrors import InputError, SimulationError
from pygom.utilR.distn import rexp, ppois, rpois

import numpy

def firstReaction(x, t, stateChangeMat, transitionFunc):
    '''
    The first reaction method
    '''
    rate = transitionFunc(x,t)
    # find our jump times
    jumpTimes = newJumpTimes(rate)
    # first jump
    minIndex = numpy.argmin(jumpTimes)
    # validate the jump times
    if jumpTimes[minIndex] == numpy.Inf:
        # if we cannot perform any more jumps
        raise SimulationError("Cannot perform any more reactions")
    else:
        newX = _updateStateWithJump(x, minIndex, stateChangeMat)
        return _checkJump(x, newX, t, jumpTimes[minIndex])

def tauLeap(x, t, stateChangeMat, reactantMat, transitionFunc, transitionMeanFunc, transitionVarFunc, epsilon=0.01):
    '''
    The Poisson :math:`\tau`-Leap
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
        maxExceed = numpy.max(1.0-numpy.array(exceedCDFArray))
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

def newJumpTimes(rates, numTransition=None):
    # if numTransition is None:
    #     numTransition = len(transition)
    
#     tau = numpy.ones(numTransition) * numpy.Inf
#     for rate in transition:
#         if rate > 0:
#             tau[i] = rexp(1, rate)
    
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
