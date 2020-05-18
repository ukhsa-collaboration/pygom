"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module with functions to perform stochastic simulation

"""
import functools

import numpy as np
import scipy.stats as st

from pygom.utilR.distn import rexp,  rpois, runif, test_seed

from ._model_errors import InputError, SimulationError
from .ode_utils import check_array_type

# Code from the cython module
from ._tau_leap import _cy_test_tau_leap_safety


def exact(x0, t0, t1, state_change_mat, transition_func,
          output_time=False, seed=None):
    """
    Stochastic simulation using an exact method starting from time
    t0 to t1 with the starting state values of x0

    Parameters
    ----------
    x0: array like
        state vector
    t0: double
        start time
    t1: double
        final time
    state_change_mat: array like
        State change matrix :math:`V_{i,j}` where :math:`i,j` represent the
        state and transition respectively.  :math:`V_{i,j}` is some
        non-zero integer such that transition :math:`j` happens means
        that state :math:`i` changes by :math:`V_{i,j}` amount
    transition_func: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition rates
    output_time: bool, optional
        defaults to False, if True then a tuple of two elements will be
        returned, else only the state vector
    seed: optional
        represents which type of seed to use.  None will defaults to the
        current global state while False will reinitialize to the initial
        global state. When seed is an integer number, it will reset the seed
        via np.random.seed.  When seed=True, then a
        :class:`np.random.RandomState` object will be used for the
        underlying random number generating process. If seed is an object
        of :class:`np.random.RandomState` then it will be used directly

    Returns
    -------
    x: array like
        state vector
    t: double
        time
    """

    x = check_array_type(x0)
    t = t0

    while t < t1:
        x_new, t_new, s = firstReaction(x, t,
                                        state_change_mat, transition_func,
                                        seed=seed)
        if s:
            x, t = x_new, t_new
        else:
            break

    if output_time:
        return x, t
    else:
        return x


def hybrid(x0, t0, t1, state_change_mat, reactant_mat,
           transition_func, transition_mean_func, transition_var_func,
           output_time=False, seed=None):
    """
    Stochastic simulation using an hybrid method that uses either the
    first reaction method or the :math:`\\tau`-leap depending on the
    size of the states and transition rates.  Starting from time
    t0 to t1 with the starting state values of x0.

    Parameters
    ----------
    x0: array like
        state vector
    t0: double
        start time
    t1: double
        final time
    state_change_mat: array like
        State change matrix :math:`V_{i,j}` where :math:`i,j` represent the
        state and transition respectively.  :math:`V_{i,j}` is some
        non-zero integer such that transition :math:`j` happens means
        that state :math:`i` changes by :math:`V_{i,j}` amount
    reactant_mat:array like
        Reactant matrix of :math:`\\lambda_{i,j}` where :math:`i,j` represents
        the index of the state and transition respectively.
        A value of 1 if state i is involved in transition j
    transition_func: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition rates
    transition_mean_func: callable
        a function that takes the input argument (x,t) and returns the
        expected transitions
    transition_var_func: callable
        a function that takes the input argument (x,t) and returns the
        variance of the transitions
    output_time: bool, optional
        defaults to False, if True then a tuple of two elements will be
        returned, else only the state vector
    seed: optional
        represents which type of seed to use.  None will defaults to the
        current global state while False will reinitialize to the initial
        global state. When seed is an integer number, it will reset the seed
        via np.random.seed.  When seed=True, then a
        :class:`np.random.RandomState` object will be used for the
        underlying random number generating process. If seed is an object
        of :class:`np.random.RandomState` then it will be used directly

    Returns
    -------
    x: array like
        state vector
    t: double
        time
    """

    x = check_array_type(x0)
    t = t0

    f = firstReaction
    while t < t1:
        if np.min(x) > 10:
            x_new, t_new, s = tauLeap(x, t,
                                      state_change_mat, reactant_mat,
                                      transition_func,
                                      transition_mean_func,
                                      transition_var_func,
                                      seed=seed)
            if s is False:
                x_new, t_new, s = f(x, t, state_change_mat,
                                    transition_func, seed=seed)
        else:
            x_new, t_new, s = f(x, t, state_change_mat,
                                transition_func, seed=seed)
        if s:
            x, t = x_new, t_new
        else:
            break

    if output_time:
        return x, t
    else:
        return x


def cle(x0, t0, t1, state_change_mat, transition_func,
        h=None, n=500, positive=True, output_time=False, seed=None):
    """
    Stochastic simulation using the CLE approximation starting from time
    t0 to t1 with the starting state values of x0.  The CLE approximation
    is performed using a simple Euler-Maruyama method with step size h.
    We assume that the input parameter transition_func provides
    :math:`f(x,t)` while the CLE is defined as
    :math:`dx = x + V*h*f(x,t) + \\sqrt(f(x,t))*Z*\\sqrt(h)
    with Z being standard normal random variables.

    Parameters
    ----------
    x0: array like
        state vector
    t0: double
        start time
    t1: double
        final time
    state_change_mat: array like
        State change matrix :math:`V_{i,j}` where :math:`i,j` represent the
        state and transition respectively.  :math:`V_{i,j}` is some
        non-zero integer such that transition :math:`j` happens means
        that state :math:`i` changes by :math:`V_{i,j}` amount
    transition_func: callable
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
        represents which type of seed to use.  None will defaults to the
        current global state while False will reinitialize to the initial
        global state. When seed is an integer number, it will reset the seed
        via np.random.seed.  When seed=True, then a
        :class:`np.random.RandomState` object will be used for the
        underlying random number generating process. If seed is an object
        of :class:`np.random.RandomState` then it will be used directly

    Returns
    -------
    x: array like
        state vector
    t: double
        time
    """

    assert isinstance(state_change_mat, np.ndarray), \
        "state_change_mat should be a np array"

    if hasattr(positive, '__iter__'):
        assert len(positive) == len(x0), \
            "an array for the input positive should have same length as x"
        assert all(isinstance(p, bool) for p in positive), \
            "elements in positive should be a bool"
        positive = np.array(positive)
    else:
        assert isinstance(positive, bool), "positive should be a bool"
        positive = np.array([positive]*len(x0))

    rvs = test_seed(seed).normal

    if h is None:
        h = (t1 - t0)/n

    x = check_array_type(x0)
    t = t0
    p = state_change_mat.shape[1]

    while t < t1:
        mu = transition_func(x, t)
        sigma = np.sqrt(mu)*rvs(0, np.sqrt(h), size=p)
        x = np.add(x, state_change_mat.dot(h*mu + sigma))
        ## We might like to put a defensive line below to stop the states
        ## going below zero.  This applies only to models where each state
        ## represent a physical count
        x[x[positive] < 0] = 0
        t += h

    if output_time:
        return x, t
    else:
        return x


def sde(x0, t0, t1, drift, diffusion, state_change_mat=None,
        h=None, n=500, positive=True, output_time=False, seed=None):
    """
    Stochastic simulation using a SDE approximation starting from time
    t0 to t1 with the starting state values of x0.  The SDE approximation
    is performed using a simple Euler-Maruyama method with step size h.
    We assume that the input parameter drift and diffusion each gives
    a function that takes in two arguments :math:`(x,t)` and computes
    the drift and diffusion.  If state_change_mat is a
    :class:`np.ndarray` then we assume that a pre-multiplication
    against the drift and diffusion is required.

    Parameters
    ----------
    x0: array like
        state vector
    t0: double
        start time
    t1: double
        final time
    drift: callable
        a function that takes the input argument (x,t) and returns the vector
        that contains the drift
    diffusion: callable
        a function that takes the input argument (x,t) and returns the vector
        that contains the diffusion
    state_change_mat: array like
        State change matrix :math:`V_{i,j}` where :math:`i,j` represent the
        state and transition respectively.  :math:`V_{i,j}` is some
        non-zero integer such that transition :math:`j` happens means
        that state :math:`i` changes by :math:`V_{i,j}` amount
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
        represents which type of seed to use.  None will defaults to the
        current global state while False will reinitialize to the initial
        global state. When seed is an integer number, it will reset the seed
        via np.random.seed.  When seed=True, then a
        :class:`np.random.RandomState` object will be used for the
        underlying random number generating process. If seed is an object
        of :class:`np.random.RandomState` then it will be used directly

    Returns
    -------
    x: array like
        state vector
    t: double
        time
    """

    if state_change_mat is not None:
        assert isinstance(state_change_mat, np.ndarray), \
            "state_change_mat should be a np array"
        p = state_change_mat.shape[1]
    else:
        p = len(drift(x0, t0))

    if hasattr(positive, '__iter__'):
        assert len(positive) == len(x0), \
            "an array for the input positive should have same length as x"
        assert all(isinstance(p, bool) for p in positive), \
            "elements in positive should be a bool"
        positive = np.array(positive)
    else:
        assert isinstance(positive, bool), "positive should be a bool"
        positive = np.array([positive]*len(x0))

    rvs = test_seed(seed).normal

    if h is None:
        h = (t1 - t0)/n

    x = check_array_type(x0)
    t = t0

    while t < t1:
        mu = h*drift(x, t)
        sigma = diffusion(x, t)*rvs(0, np.sqrt(h), size=p)
        if state_change_mat is None:
            x += mu + sigma
        else:
            x += state_change_mat.dot(mu + sigma)
        ## We might like to put a defensive line below to stop the states
        ## going below zero.  This applies only to models where each state
        ## represent a physical count
        x[x[positive]<0] = 0
        t += h

    if output_time:
        return x, t
    else:
        return x


def directReaction(x, t, state_change_mat, transition_func, seed=None):
    """
    The direct reaction method.  Same as :func:`firstReaction` for both
    input and output, only differ in internal computation
    """

    rates = transition_func(x, t)
    total_rate = sum(rates)
    jump_rate = np.cumsum(rates)

    if total_rate > 0:
        jump_time = rexp(1, total_rate, seed=seed)
        # U \sim \UnifDist[0,1]
        u = runif(1)
        target_rate = total_rate*u
        # find the index that covers the probability of jump using binary search
        transition_index = np.searchsorted(jump_rate, target_rate)
        # we can move!! move particles
        new_x = _updateStateWithJump(x, transition_index, state_change_mat)
        return _checkJump(x, new_x, t, jump_time)
    else:
        # we can't jump
        raise SimulationError("Cannot perform any more reactions")

def firstReaction(x, t, state_change_mat, transition_func, seed=None):
    """
    The first reaction method

    Parameters
    ----------
    x: array like
        state vector
    t: double
        time
    state_change_mat: array like
        State change matrix :math:`V_{i,j}` where :math:`i,j` represent the
        state and transition respectively.  :math:`V_{i,j}` is some
        non-zero integer such that transition :math:`j` happens means
        that state :math:`i` changes by :math:`V_{i,j}` amount
    transition_func: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition rates
    seed: optional
        represents which type of seed to use.  None will defaults to the
        current global state while False will reinitialize to the initial
        global state. When seed is an integer number, it will reset the seed
        via np.random.seed.  When seed=True, then a
        :class:`np.random.RandomState` object will be used for the
        underlying random number generating process. If seed is an object
        of :class:`np.random.RandomState` then it will be used directly

    Returns
    -------
    x: array like
        state vector
    t: double
        time
    success:
        if the leap was successful.  A change in both x and t if it is
        successful, no change otherwise
    """

    rates = transition_func(x, t)
    # find our jump times
    jump_times = _newJumpTimes(rates, seed=seed)
    if np.all(jump_times == np.Inf):
        return x, t, False
    # first jump
    min_index = np.argmin(jump_times)
    new_x = _updateStateWithJump(x, min_index, state_change_mat)
    return _checkJump(x, new_x, t, jump_times[min_index])


def nextReaction(x, t, state_change_mat, dependency_graph,
                 old_rates, jump_times, transition_func, seed=None):
    """
    The next reaction method
    """

    # smallest time :)
    index = np.argmin(jump_times)
    # moving state and time
    new_x = _updateStateWithJump(x, index, state_change_mat)
    t = jump_times[index]
    # recalculate the new transition matrix
    if hasattr(transition_func, '__call__'):
        rates = transition_func(x, t)
        # update the jump time
        jump_times[index] = t + rexp(1, rates[index], seed=seed)
    elif hasattr(transition_func, '__iter__'):
        rates = transition_func[index](x, t)
        jump_times[index] = t + rexp(1, rates, seed=seed)
    else:
        raise InputError("transition_func should be a single or list of callable")

    # then go through the remaining transitions
    for i, anew in enumerate(rates):
        # obviously, not the target transition as we have already fixed it
        if i != index:
            # and only if the rate has been affected by the state update
            if dependency_graph[i, index] != 0:
                aold = old_rates[i]
                if anew > 0:
                    jump_times[i] = (aold/anew)*(jump_times[i] - t) + t
                else:
                    jump_times[i] = np.Inf
        # done :)
        return new_x, t, True, rates, jump_times
    else:
        raise SimulationError("Cannot perform any more reactions")


def tauLeap(x, t, state_change_mat, reactant_mat,
            transition_func, transition_mean_func, transition_var_func,
            epsilon=0.1, seed=None):
    """
    The Poisson :math:`\\tau`-Leap

    Parameters
    ----------
    x: array like
        state vector
    t: double
        time
    state_change_mat: array like
        State change matrix :math:`V_{i,j}` where :math:`i,j` represent the
        state and transition respectively.  :math:`V_{i,j}` is some
        non-zero integer such that transition :math:`j` happens means
        that state :math:`i` changes by :math:`V_{i,j}` amount
    reactant_mat:array like
        Reactant matrix of :math:`\\lambda_{i,j}` where :math:`i,j` represents
        the index of the state and transition respectively.
        A value of 1 if state i is involved in transition j
    transition_func: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition rates
    transition_mean_func: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition mean
    transition_var_func: callable
        a function that takes the input argument (x,t) and returns the vector
        of transition variance
    epsilon: double, optional
        tolerance of the size of the jump, defaults to 0.1
    seed: optional
        represents which type of seed to use.  None will defaults to the
        current global state while False will reinitialize to the initial
        global state. When seed is an integer number, it will reset the seed
        via np.random.seed.  When seed=True, then a
        :class:`np.random.RandomState` object will be used for the
        underlying random number generating process. If seed is an object
        of :class:`np.random.RandomState` then it will be used directly

    Returns
    -------
    x: array like
        state vector
    t: double
        time
    success:
        if the leap was successful.  A change in both x and t if it is
        successful, no change otherwise
    """

    # go through the list of transitions
    rates = transition_func(x, t)

    mu = transition_mean_func(x, t)
    sigma2 = transition_var_func(x, t)
    # then we go find out the condition
    # \min_{j \in \left[1,M\right]} \{ l,r \}
    # where l = \gamma / \abs(\mu_{j}(x)) ,
    # and r = \gamma^{2} / \sigma_{j}^{2}(x)
    top = epsilon*np.sum(rates)
    try:
        l = top/abs(mu)
    except Warning:
        print("Warning as an exception")
        print(mu)
        print(x)
        print(t)
        print(rates)
    r = (top**2)/sigma2
    tau_scale = min(min(l), min(r))
    # note that the above calculation is actually very slow, because
    # we can rewrite the conditions into
    # \min \{ \min_{j \in \left[1,M\right]} l , \min_{j \in \left[1,M\right]} r \}
    # which again can be further simplified into
    # \gamma / \max_{j \in \left[1,\M\right]} \{ \abs(\mu_{j}(x),\sigma_{j}^{2} \}

    # we put in an additional safety mechanism here where we also evaluate
    # the probability that a realization exceeds the observations and further
    # decrease the time step.
    tau_scale, safe = _cy_test_tau_leap_safety(x.astype(np.float64, copy=False),
                                               reactant_mat.astype(np.int64, copy=False),
                                               rates.astype(np.float64, copy=False),
                                               float(tau_scale),
                                               float(epsilon))
    if safe is False:
        return x, t, False

    # make the jumps
    new_x = x.copy()
    for i, r in enumerate(rates):
        # realization
        try:
            jumpQuantity = rpois(1, tau_scale*r, seed=seed)
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
        new_x = _updateStateWithJump(new_x, i, state_change_mat, jumpQuantity)
        ## done moving
    return _checkJump(x, new_x, t, tau_scale)


def _test_tau_leap_safety(x, reactant_mat, rates, tau_scale, epsilon):
    """
    Additional safety test on :math:`\\tau`-leap, decrease the step size if
    the original is not small enough.  Decrease a couple of times and then
    bail out because we don't want to spend too long decreasing the
    step size until we find a suitable one.
    """
    total_rate = sum(rates)
    #reactant_mat_bin = reactant_mat == 1
    safe = False
    count = 0
    while safe is False:
        cdf_val = 1.0
        for i, r in enumerate(rates):
            xi = x[reactant_mat[:, i]]
            new_cdf = _ppois(xi, mu=tau_scale*r).min()
            if new_cdf < cdf_val:
                cdf_val = new_cdf
            #cdf_val[i * reactant_mat.shape[0] : (i * reactant_mat.shape[0]) + len(rates)] = _ppois(xi, mu=tau_scale*r)

        # the expected probability that our jump will exceed the value
        max_cdf = 1.0 - cdf_val
        # cannot allow it to exceed out epsilon
        if max_cdf > epsilon:
            tau_scale /= (max_cdf / epsilon)
        else:
            safe = True

        if tau_scale*total_rate <= 1.0 or count > 256:
            return False
        count += 1

    return tau_scale, True

#@functools.lru_cache(maxsize=2^12, typed=False)
def _ppois(q, mu=1.0):
    '''
    A cached and slightly faster and less safe version of the pygom.utilR.ppois
    function
    '''
    return st.poisson._cdf(q, mu=mu)

def _newJumpTimes(rates, seed=None):
    """
    Generate the new jump times assuming that the rates follow an exponential
    distribution
    """

    tau = [rexp(1, r, seed=seed) if r > 0 else np.Inf for r in rates]
    return np.array(tau)


def _updateStateWithJump(x, transition_index, state_change_mat, n=1.0):
    """
    Updates the states given a jump.  Makes use the state change
    matrix, and updates according to the number of times this
    transition has happened
    """
    return x + state_change_mat[:, transition_index]*n


def _checkJump(x, new_x, t, jump_time):
    failed_jump = np.any(new_x < 0)

    if failed_jump:
        # print "Illegal jump, x: %s, new x: %s" % (x, new_x)
        return x, t, False
    else:
        t += jump_time
        return new_x, t, True
