"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module with functions to perform stochastic simulation

"""

from ._model_errors import InputError, SimulationError
from pygom.utilR.distn import rexp, ppois, rpois, runif

import numpy as np
import scipy.stats

def exact(x0, t0, t1, stateChangeMat, transitionFunc,
          output_time=False, seed=None):
    '''
    Stochastic simulation using an exact method starting from time
    t0 to t1 with the starting state values of x0
    
    Parameters
    ----------
    x: array like
        state vector
    t0: double
        start time
    t1: double
        final time
    stateChangeMat: array like
        State change matrix :math:`V_{i,j}` where :math:`i,j` represent the
        state and transition respectively.  :math:`V_{i,j}` is some
        non-zero integer such that transition :math:`j` happens means
        that state :math:`i` changes by :math:`V_{i,j}` amount
    transitionFunc: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition rates
    output_time: bool, optional
        defaults to False, if True then a tuple of two elements will be
        returned, else only the state vector
    seed: optional
        represent which type of seed to use.  None or False uses the
        default seed.  When seed is an integer number, it will reset the seed
        via numpy.random.seed.  When seed=True, then a
        :class:`numpy.random.RandomState` object will be used for the
        underlying random number generating process. If seed is an
    object of :class:`numpy.random.RandomState` then it will be used directly


    Returns
    -------
    x: array like
        state vector
    t: double
        time
    '''

    x = x0
    t = t0
    if seed: seed = np.random.RandomState()
    
    while t < t1:
        x_new, t_new, s = firstReaction(x, t,
                                        stateChangeMat, transitionFunc,
                                        seed=seed)
        if s:
            if t_new > t1:
                break
            else:
                x, t = x_new, t_new
        else:
            break
        
    if output_time:
        return(x, t)
    else:
        return(x)

def cle(x0, t0, t1, stateChangeMat, transitionFunc,
        h=None, n=500, positive=True, output_time=False, seed=None):
    '''
    Stochastic simulation using the CLE approximation starting from time
    t0 to t1 with the starting state values of x0.  The CLE approximation
    is performed using a simple Euler-Maruyama method with step size h.
    We assume that the input parameter transitionFunc provides
    :math:`f(x,t)` while the CLE is defined as
    :math:`dx = x + V*h*f(x,t) + \\sqrt(f(x,t))*Z*\\sqrt(h)
    with Z being standard normal random variables.
    
    Parameters
    ----------
    x: array like
        state vector
    t0: double
        start time
    t1: double
        final time
    stateChangeMat: array like
        State change matrix :math:`V_{i,j}` where :math:`i,j` represent the
        state and transition respectively.  :math:`V_{i,j}` is some
        non-zero integer such that transition :math:`j` happens means
        that state :math:`i` changes by :math:`V_{i,j}` amount
    transitionFunc: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition rates
    h: double, optional
        step size h, defaults to None which then h = (t1 - t0)/n
    n: int, optional
        number of steps to take for the whole simulation, defaults to 500
    positive: bool or array of bool, optional
        whether the states :math:`x >= 0`.  If input is an array then the
        length should be the same as len(x)
    output_time: bool, optional
        defaults to False, if True then a tuple of two elements will be
        returned, else only the state vector
    seed: optional
        represent which type of seed to use.  None or False uses the
        default seed.  When seed is an integer number, it will reset the seed
        via numpy.random.seed.  When seed=True, then a
        :class:`numpy.random.RandomState` object will be used for the
        underlying random number generating process. If seed is an
    object of :class:`numpy.random.RandomState` then it will be used directly

    Returns
    -------
    x: array like
        state vector
    t: double
        time
    '''
    
    assert isinstance(stateChangeMat, np.ndarray), \
            "stateChangeMat should be a numpy array"
    
    if hasattr(positive, '__iter__'):
        assert len(positive) == len(x0), \
        "an array for the input positive should have same length as x"
        assert all(isinstance(p, bool) for p in positive), \
        "elements in positive should be a bool"
        positive = np.array(positive)
    else:
        assert isinstance(positive, bool), "positive should be a bool"
        positive = np.array([positive]*len(x0))

    if seed:
        rvs = np.random.RandomState().normal
    else:
        rvs = scipy.stats.norm.rvs
    
    if h is None:
        h = (t1 - t0)/n
        
    x = x0
    t = t0
    p = stateChangeMat.shape[1]

    while t < t1:
        mu = transitionFunc(x, t)
        sigma = np.sqrt(mu)*rvs(0, np.sqrt(h), size=p)
        x_new = x + stateChangeMat.dot(h*mu + sigma)
        x_new[x_new[positive]<0] = 0
        ## We might like to put a defensive line below to stop the states
        ## going below zero.  This applies only to models where each state
        ## represent a physical count
        # x_new[x_new<0] = 0
        t_new = t + h
        if t_new > t1:
            break
        else:
            x, t = x_new, t_new
        
    if output_time:
        return(x, t)
    else:
        return(x)

def directReaction(x, t, stateChangeMat, transitionFunc, seed=None):
    '''
    The direct reaction method.  Same as :func:`firstReaction` for both
    input and output, only differ in internal computation
    '''
    
    if seed: seed = np.random.RandomState()
    
    rates = transitionFunc(x,t)
    totalRate = sum(rates)
    jumpRate = np.cumsum(rates)

    if totalRate > 0:
        jumpTime = rexp(1, totalRate, seed=seed)
        # U \sim \UnifDist[0,1]
        U = runif(1)
        targetRate = totalRate*U
        # find the index that covers the probability of jump using binary search
        transitionIndex = np.searchsorted(jumpRate, targetRate)
        # we can move!! move particles
        newX = _updateStateWithJump(x, transitionIndex, stateChangeMat)
        return _checkJump(x, newX, t, jumpTime)
    else:
        # we can't jump
        raise SimulationError("Cannot perform any more reactions")
        
def firstReaction(x, t, stateChangeMat, transitionFunc, seed=None):
    '''
    The first reaction method
    
    Parameters
    ----------
    x: array like
        state vector
    t: double
        time
    stateChangeMat: array like
        State change matrix :math:`V_{i,j}` where :math:`i,j` represent the
        state and transition respectively.  :math:`V_{i,j}` is some
        non-zero integer such that transition :math:`j` happens means
        that state :math:`i` changes by :math:`V_{i,j}` amount
    transitionFunc: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition rates
    seed: optional
        represent which type of seed to use.  None or False uses the
        default seed.  When seed is an integer number, it will reset the seed
        via numpy.random.seed.  When seed=True, then a
        :class:`numpy.random.RandomState` object will be used for the
        underlying random number generating process. If seed is an
    object of :class:`numpy.random.RandomState` then it will be used directly

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
    
    if seed: seed = np.random.RandomState()
    rates = transitionFunc(x,t)
    # find our jump times
    jumpTimes = _newJumpTimes(rates, seed=seed)
    if np.all(jumpTimes == np.Inf):
        return(x, t, False)
    # first jump
    minIndex = np.argmin(jumpTimes)
    # validate the jump times
    if jumpTimes[minIndex] == np.Inf:
        # if we cannot perform any more jumps
        raise SimulationError("Cannot perform any more reactions")
    else:
        newX = _updateStateWithJump(x, minIndex, stateChangeMat)
        return _checkJump(x, newX, t, jumpTimes[minIndex])

def nextReaction(x, t, stateChangeMat, dependencyGraph,
                 oldRates, jumpTimes, transitionFunc, seed=None):
    '''
    The next reaction method
    '''

    if seed: seed = np.random.RandomState()
    # smallest time :)
    index = np.argmin(jumpTimes)
    # moving state and time
    newX = _updateStateWithJump(x, index, stateChangeMat)
    t = jumpTimes[index]
    # recalculate the new transition matrix
    if hasattr(transitionFunc, '__call__'):
        rates = transitionFunc(x,t)
        # update the jump time
        jumpTimes[index] = t + rexp(1, rates[index], seed=seed)
    elif hasattr(transitionFunc, '__iter__'):
        jumpTimes[index] = t + rexp(1, transitionFunc[index](x, t), seed=seed)
    else:
        raise InputError("transitionFunc should be a single or list of callable")

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
                    jumpTimes[i] = np.Inf
        # done :)
        return newX, t, True, rates, jumpTimes
    else:
        raise SimulationError("Cannot perform any more reactions")


def tauLeap(x, t, stateChangeMat, reactantMat,
            transitionFunc, transitionMeanFunc, transitionVarFunc,
            epsilon=0.1, seed=None):
    '''
    The Poisson :math:`\\tau`-Leap
    
    Parameters
    ----------
    x: array like
        state vector
    t: double
        time
    stateChangeMat: array like
        State change matrix :math:`V_{i,j}` where :math:`i,j` represent the
        state and transition respectively.  :math:`V_{i,j}` is some
        non-zero integer such that transition :math:`j` happens means
        that state :math:`i` changes by :math:`V_{i,j}` amount
    reactantMat:array like
        Reactant matrix of :math:`\\lambda_{i,j}` where :math:`i,j` represents
        the index of the state and transition respectively.
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
        tolerance of the size of the jump, defaults to 0.1
    
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
    
    if seed: seed = np.random.RandomState()
    # go through the list of transitions
    rates = transitionFunc(x,t)
    totalRate = sum(rates)

    safeToJump = False

    mu = transitionMeanFunc(x, t)
    sigma2 = transitionVarFunc(x, t)
    # then we go find out the condition
    # \min_{j \in \left[1,M\right]} \{ l,r \}
    # where l = \gamma / \abs(\mu_{j}(x)) ,
    # and r = \gamma^{2} / \sigma_{j}^{2}(x)
    top = epsilon*totalRate
    l = top/abs(mu)
    r = (top**2)/sigma2
    tauScale = min(min(l), min(r))
    # note that the above calculation is actually very slow, because
    # we can rewrite the conditions into
    # \min \{ \min_{j \in \left[1,M\right]} l , \min_{j \in \left[1,M\right]} r \}
    # which again can be further simplified into
    # \gamma / \max_{j \in \left[1,\M\right]} \{ \abs(\mu_{j}(x),\sigma_{j}^{2} \}

    # we put in an additional safety mechanism here where we also evaluate
    # the probability that a realization exceeds the observations and further
    # decrease the time step.
#     print "tauScale Orig = %s " % tauScale
#     print len(rates)
#     print reactantMat
    while safeToJump == False:
        exceedCDFArray = list()
        for i, r in enumerate(rates):
            activeX = x[reactantMat[:,i]]
            for xi in activeX:
                exceedCDFArray.append(ppois(xi, tauScale*r))

        # the expected probability that our jump will exceed the value
        maxExceed = np.max(1.0 - np.array(exceedCDFArray))
        # cannot allow it to exceed out epsilon
        if maxExceed > epsilon:
            tauScale /= 2.0
        else:
            safeToJump = True
        if tauScale*totalRate <= 1.0:
            return x, t, False
    ## end while safeToJump==False

    # print tauScale
    # print rates

    # make the jumps
    newX = x.copy()
    for i, r in enumerate(rates):
        # realization
        try:
            jumpQuantity = rpois(1, tauScale*r, seed=seed)
        except Exception as e:
#             print tauScale, r
#             print "l = %s " % l
#             print "r = %s " % (top**2 / sigma2)
#             print "top = %s " % top
#             print "min (l, r) = (%s, %s)"  % (min(l), min(top**2 / sigma2))
#             print "tauScale = %s" % tauScale
#             print "exceed %s " % len(exceedCDFArray)
#             print "mu = %s " % mu
#             print "sigma2 = %s " % sigma2 
            raise e
            
        # print jumpQuantity
        # move the particles!
        newX = _updateStateWithJump(newX, i, stateChangeMat, jumpQuantity)
        ## done moving
    return _checkJump(x, newX, t, tauScale)

def _newJumpTimes(rates, seed=None):
    '''
    Generate the new jump times assuming that the rates follow an exponential
    distribution
    '''   
    tau = [rexp(1, r, seed=seed) if r > 0 else np.Inf for r in rates]
    return np.array(tau)

def _updateStateWithJump(x, transitionIndex, stateChangeMat, n=1.0):
    '''
    Updates the states given a jump.  Makes use the state change
    matrix, and updates according to the number of times this
    transition has happened
    '''
    return x + stateChangeMat[:,transitionIndex]*n
    
def _checkJump(x, newX, t, jumpTime):
    failedJump = np.any(newX < 0)

    if failedJump:
        # print "Illegal jump, x: %s, new x: %s" % (x, newX)
        return x, t, False
    else:
        t += jumpTime
        return newX, t, True
