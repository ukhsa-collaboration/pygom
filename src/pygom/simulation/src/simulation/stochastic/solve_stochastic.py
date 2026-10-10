"""
Stochastic solver function
"""

import numpy as np
import time
from .factory import make_stepper
from .config import TauConfig, ExactConfig
from ..data_classes import Output, PerformanceMetrics, TimeSeriesResult, EventSeriesResult
from ..post_processing import *
from ..stochastic.config_api import build_config


def solve_stochastic(
        event_rates,
        stoichiometry_matrix,
        t_span,
        y0,
        y_min,
        y_max,
        method,
        t_eval=None,
        rng=None,
        proceed_if_rates_zero=False,
        perf=False,
        **options):
    """
    Compute a stochastic solution to an initial value problem defined by a continuous-time Markov chain.

    Parameters
    ----------
    event_rates : callable
        Function of time and state, f(t, y), which returns an array of the
        rates of occurance of each event.
    stoichiometry_matrix: numpy.ndarray
        Integer-valued matrix where element [i, j] represents the change in state i
        resulting from an occurance of event j.
        Note: May also be referred to as the "state change matrix" or "reaction matrix".
    t_span : 2-member sequence
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. Both t0 and tf must be floats
        or values interpretable by the float conversion function.
    y0 : numpy.ndarray
        Initial (integer) state values
    y_min: numpy.ndarray
        Minimum allowed state values
    y_max: numpy.ndarray
        Maximum allowed state values
    method : string
        Stochastic stepping method to use. Due to the fundamentally different
        approaches of exact and tau-leaping methods, and the difficulty of
        automatically selecting step sizes for the latter, the user must
        explicitly specify the method and any relevant optional parameters.

            * 'fixed_tau': Tau leap with fixed step size.
            * 'cao2006': Adaptive tau leap, with step size computed using the
              algorithm of Cao et al (2006).
            * 'direct': Exact solution via direct reaction method.
            * 'first_reaction': Exact solution via first reaction method.

        It is recommended to use the 'fixed_tau' method, specifying the step size
        via the option 'tau'. The appropriate step size should typically be a
        timescale shorter than the fastest timescale of interest in the system.

    t_eval : array_like or None, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points selected by the solver.
    seed : int
        Number used to initialise random number generator
    proceed_if_rates_zero : bool
        If True, a tau leap method will keep going until tf, even if the event rates are zero.
        If False, simulation will prematurely end.
    perf : bool
        If True, performance metrics will be recorded and appended to simulation output
    **options
        Options passed to a chosen solver. All options available for already
        implemented solvers are listed below. Only the tau leaping methods necessitate
        these extra options.
    checker : str
        Calculation performed to check if a proposed tau-leap step is invalid.

            * 'forbidden_reaction' (default): Check that each individual reactions do not occur
              more than it may due to state populations (this will conserve population)
            * 'forbidden_state': Check that states remain within their permitted
              range of values
            * 'none': No checks (not recommended)
        
    refiner : str
        Calculation performed pre-jump to anticipate if step is likley to generate
        illegal state values
        
            * 'none' (default): No precaution
            * 'prob': Probabilistic method. If probability of illegal step is greater than
              acceptable_prob_misstep, reduce step size by amount between factor_min and
              factor_max for max_retries attempts.

    refiner_opts : dict
        max_retries : int, default = 10
        acceptable_prob_misstep : float, default = 0.05
        factor_min : float, default = 0.1
        factor_max : float, default = 0.9

    retry_max : int, default = 10
        In the event of illegal step, how many times to retry before aborting simulation
    tau_rescale : float, default = 0.5
        In the event of illegal step, by what factor to reduce step size before retrying.
    tau : float
        Time step size for 'fixed_tau' method
    epsilon: float
        Adaptive step size parameter from Cao 2006.
    transition_mean_func: callable
        Equation (8a) of Cao 2006
    transition_var_func: callable
        Equation (8b) of Cao 2006
    """

    #########################
    # Type checking arguments
    #########################
    # ----
    # Time
    # ---- 
    # (Use scipy.integrate.solve_ode framework to initialise)
    t0, tf = map(float, t_span)
    post_process = False
    if t_eval is not None:
        post_process = True
        t_eval = np.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        # For now assume only possible to stochastically intergrate forwards in time
        d = np.diff(t_eval)
        if tf > t0 and np.any(d <= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")
        
    # -------------
    # Stoichiometry
    # -------------
    stoichiometry_dims = stoichiometry_matrix.shape
    if len(stoichiometry_dims) != 2:
        raise ValueError(f"Reaction matrix must be 2 dimensional, instead received {len(stoichiometry_dims)} dimensions")

    if not np.issubdtype(stoichiometry_matrix.dtype, np.integer):
        raise ValueError(f"Reaction matrix must contain only integer dtype")
    
    n_state, n_events = stoichiometry_dims

    # --------------------------------------
    # States (initial conditions and limits)
    # --------------------------------------
    if not isinstance(y0, np.ndarray):
        raise TypeError("y0 must be a numpy array")
    if y0.ndim != 1:
        raise ValueError("y0 must be 1 dimensional")
    if not np.issubdtype(y0.dtype, np.integer):
        raise TypeError("y0 must have integer dtype")
    if len(y0) != n_state:
        raise ValueError(f"y0 ({len(y0)}) and reaction matrix ({n_state}) disagree on number of states")

    if not isinstance(y_min, np.ndarray):
        raise TypeError("y_min must be a numpy array")
    if y_min.ndim != 1:
        raise ValueError("y_min must be 1 dimensional")
    if not np.issubdtype(y_min.dtype, np.integer):
        raise TypeError("y_min must have integer dtype")
    if len(y_min) != n_state:
        raise ValueError(f"y_min ({len(y_min)}) and reaction matrix ({n_state}) disagree on number of states")

    if not isinstance(y_max, np.ndarray):
        raise TypeError("y_max must be a numpy array")
    if y_max.ndim != 1:
        raise ValueError("y_max must be 1 dimensional")
    # if not np.issubdtype(y_max.dtype, np.integer):
    #     raise TypeError("y_max must have integer dtype")
    if len(y_max) != n_state:
        raise ValueError(f"y_max ({len(y_max)}) and reaction matrix ({n_state}) disagree on number of states")

    ####################
    # Build time stepper
    ####################
    config = build_config(method, **options)

    stepper = make_stepper(
        config=config,
        event_rates=event_rates,
        stoichiometry_matrix=stoichiometry_matrix,
        y_min=y_min,
        y_max=y_max,
        proceed_if_rates_zero=proceed_if_rates_zero,
        rng=rng
    )

    ################
    # Run simulation
    ################
    t = t0
    y = np.array(y0, dtype=np.int64)        # For now, only allow integer states

    if perf:
        start_time = time.perf_counter()
        start_cpu = time.process_time()

    if isinstance(config, ExactConfig):
        extra_info_name = "event_idx"
        build_result = lambda t, y, extra: EventSeriesResult(t=t, y=y, event_id=extra)
    elif isinstance(config, TauConfig):
        extra_info_name = "event_counts"
        build_result = lambda t, y, extra: TimeSeriesResult(t=t, y=y, event_counts=extra)
    else:
        raise ValueError('Solver config is not recognised')

    # Initialise output lists
    ys = [y.copy()]
    ts = [t]
    extra_info = []

    # Main loop
    while t < tf:
        new_state = stepper.take_step(t, y)

        if new_state.end_sim:
            break

        y = new_state.y_new.copy()
        t = new_state.t_new

        ys.append(y)
        ts.append(t)
        extra_info.append(getattr(new_state, extra_info_name))

    y = np.array(ys)
    t = np.array(ts)

    sol = build_result(t, y, np.array(extra_info))
    result = Output(result=sol, config=config, diag=stepper.diag)

    if perf:
        wall_time_seconds = time.perf_counter() - start_time
        cpu_time_seconds = time.process_time() - start_cpu

        perf = {}
        perf['wall_time_seconds'] = wall_time_seconds
        perf['cpu_time_seconds'] = cpu_time_seconds

        result.performance = PerformanceMetrics(**perf)

    #################################
    # Post process simulation results
    ################################# event_ids, event_times, target_times, n_events
    if post_process:
        if isinstance(result.result, TimeSeriesResult):
            y = interpolate_state_at_times(t=result.result.t, y=result.result.y, target_time=t_eval)
            event_counts = change_bins(event_counts=result.result.event_counts, old_breaks=result.result.t, new_breaks=t_eval)
            result.result = TimeSeriesResult(t=t_eval, y=y, event_counts=event_counts)

        elif isinstance(result.result, EventSeriesResult):
            y = extract_state_at_target_times(t=result.result.t, y=result.result.y, target_time=t_eval)
            event_counts = bin_events(event_ids=result.result.event_id, event_times=result.result.t[1:], target_times=t_eval, n_events=n_events) # t=0 is not an event
            result.result = TimeSeriesResult(t=t_eval, y=y, event_counts=event_counts)

    return result