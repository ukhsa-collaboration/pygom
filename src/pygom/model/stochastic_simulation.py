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


def hybrid(x0, t0, t1, state_change_mat,
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
                                      state_change_mat,
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

def firstReaction(x, x_lims, t, state_change_mat, transition_func, seed=None):
    """
    The first reaction method

    Parameters
    ----------
    x: array like
        state vector
    x_lims: array like
        list of length 2 lists giving [max, min] limits for each state
        can be [None, None] if no limits.
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
    t_new: double
        New timepoint
    jump_times: double
        timestep
    x_new: array like
        new state
    jumps: array like
        list giving identity of the transition which fired in the timestep
    success:
        if the leap was successful (where success is defined as all x[i] falling within x_lims[i])
        Results in a change in both x and t if it is successful, no change otherwise.
    """

    rates = transition_func(x, t)
    # For now we assume when all transition rates are zero, further simulation is not necessary.
    if all(rates==0):
        return 0, 0, 0, 0, False

    # find our jump times
    jump_times = _newJumpTimes(rates, seed=seed)
    if np.all(jump_times == np.inf):
        return x, t, False
    # first jump
    min_index = np.argmin(jump_times)
    new_x = _updateStateWithJump(x, min_index, state_change_mat)

    # record which state the jump was in
    jumps=[0]*len(rates)
    jumps[min_index]=1

    return _checkJump(x, new_x, x_lims, t, jump_times[min_index], jumps)

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
                    jump_times[i] = np.inf
        # done :)
        return new_x, t, True, rates, jump_times
    else:
        raise SimulationError("Cannot perform any more reactions")


def tauLeap(x,
            x_lims,
            t,
            state_change_mat,
            reactant_mat,
            transition_func,
            transition_mean_func,
            transition_var_func,
            isStochastic,
            epsilon=0.03,
            seed=None,
            pre_tau=None):
    """
    The Poisson :math:`\\tau`-Leap. Calculates the appropriate time step
    and then moves the system forwards by this amount. The time step
    calculation involves two stages: first, a calculation of the step size
    according to the timescales of the system and second, a refinement of
    this if necessary taking into account the risk of states having negative
    populations.

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
        tolerance of the size of the jump, defaults to 0.03 as recommended by
        Cao et al. https://doi.org/10.1063/1.2159468
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
    t_new: double
        New timepoint
    jump_times: double
        timestep
    x_new: array like
        new state
    jumps: array like
        transition values
    success:
        if the leap was successful (where success is defined as all x[i] falling within x_lims[i])
        Results in a change in both x and t if it is successful, no change otherwise.
    """

    rates = transition_func(x, t)
    # For now we assume when all transition rates are zero, further simulation is not necessary.
    if all(rates==0):
        return 0, 0, 0, 0, False

    # Step 1: Calculate step size due to timescales of the system
    # TODO: add in minimum stepsize condition from Cao equations (11)-(13)

    # If no timestep specified, do adaptive tau leap.
    if pre_tau==None:
        tau_scale=_get_adaptive_tau_step(x,
                                          t,
                                          rates,
                                          transition_mean_func,
                                          transition_var_func,
                                          epsilon=epsilon)
    else:
        tau_scale=pre_tau

    # Step 2: we put in an additional safety mechanism here where we also evaluate
    #         the probability that a realization exceeds the observations and further
    #         decrease the time step.
    # TODO: consider how Cao implemented this

    loss_mat=reactant_mat.copy()
    loss_mat[loss_mat==1]=0
    loss_mat[loss_mat==-1]=1
    
    tau_scale, safe = _cy_test_tau_leap_safety(x.astype(np.float64, copy=False),
                                               loss_mat.astype(np.int64, copy=False),
                                               rates.astype(np.float64, copy=False),
                                               float(tau_scale),
                                               float(epsilon))
    if safe is False:
        return x, t, False

    # containers for output
    new_x = x.copy()                # updated state populations
    jumps=[0]*len(rates)            # transitions

    # take deterministic or stochastic step-
    for i, r in enumerate(rates):
        if isStochastic[i]==False:
            jumpQuantity = tau_scale*r
        else:
            jumpQuantity = rpois(1, tau_scale*r, seed=seed)

        jumps[i]=jumpQuantity
        new_x = _updateStateWithJump(new_x, i, state_change_mat, jumpQuantity)

    return  _checkJump(x, new_x, x_lims, t, tau_scale, jumps)

# TODO: Check speed performance of this and see if cython version necessary
def _get_adaptive_tau_step(x,
                           t,
                           rates,
                           transition_mean_func,
                           transition_var_func,
                           epsilon):
    
    """
    Adaptive :math:`\\tau`-Leap calculator

    Calculates the time step size such that, given the current state of the
    system, (x,t), no propensity function, a[i](x,t), (where i denotes the transition)
    changes appreciably over the course of the time step.

    Parameters
    ----------
    x: array like
        state vector
    t: double
        time
    rates: array like
        values of rates of each transition, i (i.e. evaluation of the propensity
        functions a[i](x,t))
    transition_mean_func: callable
        a function that takes the input argument (x,t) and returns the vector
        of the mean expected rates of change of the a[i] with time
    transition_var_func: callable
        a function that takes the input argument (x,t) and returns the vector
        of the variance of the expected rates of change of the a[i] with time
    epsilon: double, optional
        tolerance of the size of the jump,  (0.03 is recommended by
        Cao et al. https://doi.org/10.1063/1.2159468)

    Returns
    -------
    tau_scale: double
        time step which is short enough such that system equations don't change
        appreciably
    """

    mu = transition_mean_func(x, t)     # mu[i]*dt = mean of expected change in a[i] in dt
    sigma2 = transition_var_func(x, t)  # sqrt(sigma2*dt) = standard deviation of expected change in a[i] in dt

    # Some rates could be 0. Remove them to avoid dividing by 0 later on.
    mu=mu[mu!=0]
    sigma2=sigma2[sigma2!=0]

    if mu.size==0 and sigma2.size==0:
        # These values could be 0 if e.g. the epidemic is over and nothing
        # is happening anymore. The function needs to return something, though might be
        # better end the simulation (assuming nothing will bring the system
        # back to life, e.g. external introductions or immune waning).
        # For now default to step size of 1, just to complete the run.
        return np.float64(1)

    else:
        # Equations (7)-(9) from Cao et al. https://doi.org/10.1063/1.2159468
        # (publicly available copy https://people.cs.vt.edu/~ycao/publication/newstepsize.pdf)
        bound = epsilon*np.sum(rates)
        if mu.size==0:
            tau_scale = min((bound**2)/sigma2)
        elif sigma2.size==0:
            tau_scale = min(bound/abs(mu))
        else:
            tau_scale_mu=min(bound/abs(mu))
            tau_scale_sig=min((bound**2)/sigma2)
            tau_scale=min(tau_scale_mu, tau_scale_sig)
    
    return tau_scale

    # TODO: note that the above calculation is actually very slow, because
    # we can rewrite the conditions into
    # \min \{ \min_{j \in \left[1,M\right]} l , \min_{j \in \left[1,M\right]} r \}
    # which again can be further simplified into
    # \gamma / \max_{j \in \left[1,\M\right]} \{ \abs(\mu_{j}(x),\sigma_{j}^{2} \}
    # where l= bound/abs(mu) and r=(bound**2)/sigma2

# TODO: Check speed performance of this vs cython version
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

    tau = [rexp(1, r, seed=seed) if r > 0 else np.inf for r in rates]
    return np.array(tau)


def _updateStateWithJump(x, transition_index, state_change_mat, n=1.0):
    """
    Updates the states given a jump.  Makes use the state change
    matrix, and updates according to the number of times this
    transition has happened
    """
    return x + state_change_mat[:, transition_index]*n


# def _checkJump(new_x, x_lims):
#     # Check if new values fall outside of limits
#     failed_jump=False
#     for i, x_lim in enumerate(x_lims):
#         if x_lim != [None, None]:
#             x_min=x_lim[0]
#             x_max=x_lim[1]
#             if x_min is None:
#                 if new_x[i]>x_max:
#                     failed_jump=True
#             elif x_max is None:
#                 if new_x[i]<x_min:
#                     failed_jump=True
#             else:
#                 if new_x[i]<x_min or new_x[i]>x_max:
#                     failed_jump=True

#     return failed_jump

def _checkJump(x, x_new, x_lims, t, jump_time, jumps):
    # Check that new values fall within limits and increment time

    failed_jump=False
    for i, x_lim in enumerate(x_lims):
        if x_lim != [None, None]:
            x_min=x_lim[0]
            x_max=x_lim[1]
            if x_min is None:
                if x_new[i]>x_max:
                    failed_jump=True
            elif x_max is None:
                if x_new[i]<x_min:
                    failed_jump=True
            else:
                if x_new[i]<x_min or x_new[i]>x_max:
                    failed_jump=True

    if failed_jump:
        print("Illegal jump, x: %s, new x: %s" % (x, x_new))
        success=False
        x_new=x
        t_new=t
    else:
        success=True
        t_new = t + jump_time
    
    return t_new, jump_time, x_new, jumps, success
