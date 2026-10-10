import numpy as np
from abc import ABC, abstractmethod
import logging

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

# ============================================================
# Stochastic jumper base class
# ============================================================
class StochasticLeap(ABC):
    def __init__(self, transition_func, state_change_mat, mins, maxs, proceed_until_end):
        self.transition_func = transition_func
        self.state_change_mat = state_change_mat
        self.mins = mins
        self.maxs = maxs  
        self.proceed_until_end = proceed_until_end
        self.tau = 1         # in case of rates going to zero, first reaction needs something to move it on.

    def take_step(self, x, t):
        """Called by user. Returns (x_new, t_new)."""

        # Calculte functions only once (we might need to retry, so don't want to recalculate these)
        rates = self.transition_func(x, t)
        if np.all(rates == 0):
            return self._zero_rate_behavior(x, t, rates)
        if np.any(rates < 0):
            raise RuntimeError("Negative reaction rates encountered")
        changes = self.state_change_mat(x, t)

        return self._make_jump(x, t, rates, changes)

    @abstractmethod
    def _propose_jump(self, x, t, rates, changes, *args):
        """Return (jumps, dt)."""
        pass

    def _no_refine_tau(self, tau, thresholds, rates):
        return tau

    def _get_new_x(self, x, t, changes, jumps):
        return x + changes @ jumps

    def _check_jump_overall(self, x, t, changes, jumps, dt):
        """Check boundaries after applying jumps."""
        x_new = self._get_new_x(x, t, changes, jumps)

        if np.any(x_new < self.mins) or np.any(x_new > self.maxs):
            return x, t, False  # failed jump

        return x_new, t + dt, True

    def _zero_rate_behavior(self, x, t, rates):
        if self.proceed_until_end:
            # advance by tau, rates might be non zero later
            return self._successful_jump(x, t+self.tau, np.zeros_like(rates))
        else:
            # end simulation
            return x, t, np.zeros_like(rates), True

    def _successful_jump(self, x_new, t_new, jumps):
        return x_new, t_new, jumps, False

    @abstractmethod
    def _make_jump(self, x, t, rates, changes):
        pass

# ============================================================
# Tau Leap
# ============================================================
class TauLeap(StochasticLeap):
    def __init__(self, transition_func, state_change_mat, mins, maxs, proceed_until_end,
                 tau,
                 scale=0.5,
                 retry_max=10,
                 epsilon=0.01,
                 max_iter=258,
                 method_check="per_reaction",
                 method_precaution="none"):
        super().__init__(transition_func, state_change_mat, mins, maxs, proceed_until_end)

        if tau<=0:
            raise ValueError("Tau must be non zero and positive")

        CHECKERS = ["per_reaction", "overall"]
        if method_check not in CHECKERS:
            raise ValueError(f"Unknown method '{method_check}'. Options: {CHECKERS}")

        PRECAUTIONS = ["adaptive", "none"]
        if method_precaution not in PRECAUTIONS:
            raise ValueError(f"Unknown method '{method_precaution}'. Options: {PRECAUTIONS}")

        # -----------------------------------------------------------------------
        # Early binding dispatch
        # Avoid branching at runtime and declare methods now

        # Post proposed jump checks
        if method_check == "per_reaction":
            self._check_jump = lambda x, t, jumps, dt, changes, thresholds: \
                self._check_jump_individual(x, t, jumps, dt, changes, thresholds)
        elif method_check == "overall":
            self._check_jump = lambda x, t, jumps, dt, changes, thresholds: \
                self._check_jump_overall(x, t, jumps, dt, changes)

        # Pre jump tau refinements
        if method_precaution == "adaptive":
            self._update_tau = self._refine_tau
        elif method_precaution == "none":
            self._update_tau = self._no_refine_tau
        # -----------------------------------------------------------------------

        self.tau = tau
        self.scale = scale
        self.retry_max = retry_max
        self.allows_retry = True
        self.failed_step_count = 0
        self.target = np.log(1 - epsilon)
        self.max_iter = max_iter

    def _propose_jump(self, x, t, tau, rates):
        # Poisson number of jumps for each reaction channel
        jumps = np.random.poisson(rates * tau)
        return jumps, tau
    
    def _threshold_jumps(self, x, t, changes):
        # How far between current state and value limits
        min_margin = x - self.mins
        max_margin = self.maxs - x

        # Threshold number of times each reaction can occur:
        # for going below min limits
        thresholds_min = np.where(changes < 0, np.floor(min_margin[:,None] / -changes), np.inf)
        # for going above max limits
        thresholds_max = np.where(changes > 0, np.floor(max_margin[:,None] / changes), np.inf)
        # overall:
        return np.min(np.minimum(thresholds_min, thresholds_max), axis=0) 

    def _prob_illegal_jump(self, tau, thresholds, rates):
        means = rates * tau  # expected number of events per reaction
        log_p_total = poisson.logcdf(thresholds, means).sum()  # Use logcdf for numerical stability

        return log_p_total      # p_fail = exp( 1 - log_p_total )

    def _refine_tau(self, tau, thresholds, rates):
        for _ in range(self.max_iter):
            
            log_p_total = self._prob_illegal_jump(tau, thresholds, rates)

            if log_p_total >= self.target:
                return tau

            # ratio < 1 when tau is too big
            factor = self.target / log_p_total

            # prevent tau increases and ensure reasonable shrink
            factor = np.clip(factor, 0.1, 0.9)
            tau *= factor
            
        raise RuntimeError(f"No safe tau found")

    def _check_jump_individual(self, x, t, jumps, dt, changes, thresholds):
        """Check jumps don't occur more than individial limits"""
        if np.any(jumps > thresholds):
            return x, t, False  # failed jump

        x_new = self._get_new_x(x, t, changes, jumps)

        return x_new, t + dt, True

    def _make_jump(self, x, t, rates, changes):
        tau = self.tau

        thresholds = self._threshold_jumps(x, t, changes)
        tau = self._update_tau(tau, thresholds, rates)

        for _i in range(self.retry_max + 1):
            jumps, dt = self._propose_jump(x, t, tau, rates)
            x_new, t_new, success = self._check_jump(x, t, jumps, dt, changes, thresholds)

            if success:
                return self._successful_jump(x_new, t_new, jumps)

            # illegal value encountered, retry with half tau
            self.failed_step_count += 1       # log of total retries might be useful
            tau *= self.scale
        raise RuntimeError(f"Forbidden values still encountered after {self.retry_max} attempts")



# ============================================================
# First Reaction Method
# ============================================================
class FirstReaction(StochasticLeap):
    def __init__(self, transition_func, state_change_mat, mins, maxs, proceed_until_end):
        super().__init__(transition_func, state_change_mat, mins, maxs, proceed_until_end)

    def _propose_jump(self, x, t):
        jump_times = np.random.exponential(1.0 / self.rates)

        # find reaction with smallest time
        k = np.argmin(jump_times)

        jumps = np.zeros_like(self.rates, dtype=int)
        jumps[k] = 1
        dt = jump_times[k]

        return jumps, dt
    
    def _make_jump(self, x, t):
        jumps, dt = self._propose_jump(x, t)
        x_new, t_new, success = self._check_jump(x, t, jumps, dt)

        if success:
            return self._successful_jump(x_new, t_new, jumps)
        else:
            raise RuntimeError(f"Forbidden values still encountered")

# ============================================================
# Direct Reaction Method
# ============================================================
class DirectReaction(StochasticLeap):
    def __init__(self, transition_func, state_change_mat, mins, maxs, proceed_until_end):
        super().__init__(transition_func, state_change_mat, mins, maxs, proceed_until_end)
    
    def _propose_jump(self, x, t):
        total_rate = self.rates.sum()
        transition_index = np.random.choice(len(self.rates), p=self.rates / total_rate)

        jumps = np.zeros(len(self.rates), dtype=np.int8)
        jumps[transition_index] = 1

        dt = np.random.exponential(1 / total_rate)

        return jumps, dt

    def _make_jump(self, x, t):
        jumps, dt = self._propose_jump(x, t)
        x_new, t_new, success = self._check_jump(x, t, jumps, dt)

        if success:
            return self._successful_jump(x_new, t_new, jumps)
        else:
            raise RuntimeError(f"Forbidden values still encountered")


##########################
# Combinations of steppers
##########################

STOCHASTIC_STEPPER_REGISTRY = {
    "tau": TauLeap,
    "first": FirstReaction,
    "direct": DirectReaction,
}

def solve(
    x0, 
    t_max,
    ode_eqns,
    state_change_mat,
    mins,
    maxs,
    method="tau",
    t0=0.0,
    proceed_until_end=True,
    **kwargs   # to pass extra params to specific steppers
):
    # Set up stepper
    if method not in STOCHASTIC_STEPPER_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. Options: {list(STOCHASTIC_STEPPER_REGISTRY.keys())}")
    StepperClass = STOCHASTIC_STEPPER_REGISTRY[method]
    stepper = StepperClass(ode_eqns, state_change_mat, mins, maxs, proceed_until_end, **kwargs)

    # Initial conditions
    t = t0
    x = np.array(x0, dtype=float)

    # Initialise output lists
    xs = [x.copy()]
    ts = [t]
    jumps = []

    # Main loop
    while t < t_max:
        x, t, jump, end_sim = stepper.take_step(x, t)

        if end_sim:
            break

        xs.append(x.copy())
        ts.append(t)
        jumps.append(jump)

    return np.array(ts), np.array(xs), np.array(jumps), stepper.failed_step_count


#################
# Post processing
#################

def bin_jumps(t, jumps, jump_names, target_time):

    n_trans = jumps.shape[1]
    mid = 0.5 * (t[:-1] + t[1:])

    # empty matrix to receive scaled data = (new timepoints x n_trans)
    # minus one because we are looking at jumps which occur betwen timepoints
    # (e.g. 2 timepoints =1 jump, 10 timepoints =9)
    jump_bin = np.zeros((len(target_time)-1, n_trans))

    # if exact, each point corresponds to a transitions and has weight 1.
    for i in range(n_trans):
        hist, _ = np.histogram(mid, bins=target_time, weights=jumps[:,i])
        jump_bin[:, i] = hist

    df = pd.DataFrame({'t': target_time[:-1]})
    df[jump_names] = jump_bin
    
    return df

def extract_state_at_time(t, x, state_names, target_time):
    
    # Find insertion positions
    idx = np.searchsorted(t, target_time, side="right") - 1

    # Clamp to range [0, len(t)-1]
    idx = np.clip(idx, 0, len(t)-1)

    df = pd.DataFrame({'t': target_time})
    df[state_names] = x[idx]
    
    return df

def interpolate_state(t, x, state_names, target_time):
    n_state = x.shape[1]

    x_interp = np.zeros((len(target_time), n_state))  # empty matrix to receive scaled data = (new timepoints x vars)

    # linearly interpolate to new timepoints
    for i in range(n_state):
        x_interp[:, i]=np.interp(target_time, t, x[:,i])

    df = pd.DataFrame({'t': target_time})
    df[state_names] = x_interp
    
    return df

