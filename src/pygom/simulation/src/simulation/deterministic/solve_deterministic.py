import time
import numpy as np

from .integrator import DeterministicSolver
from ..data_classes import Output, PerformanceMetrics, TimeSeriesResult

from ..data_classes import SolverConfig

from dataclasses import dataclass

@dataclass
class DeterministicConfig(SolverConfig):
    method = "Deterministic"

def solve_deterministic(
        ode_eqns,
        event_rates,
        n_state,
        n_event,
        t_span,
        y0,
        jac_ode=None,
        jac_events=None,
        method='RK45',
        t_eval=None,
        dense_output=False,
        events=None,
        vectorized=False,
        args=None,
        perf=False,
        **options):
    
    """
    Solve the deterministic system by augmenting the state ODEs with event
    occurrence variables. This provides prevalence through the state variables
    and incidence through the event variables. This is analogous to recording
    event counts in the stochastic case.

    This is basically a wrapper of scipy.integrate.solve_ivp. The main
    contribution of this function is to automate the augmentation.

    ode_eqns : callable
        Function of time and state, f(t, y), which returns an array of the
        rates of change of each state
    event_rates : callable
        Function of time and state, f(t, y), which returns an array of the
        rates of occurance of each event
    n_state : int
        Number of states
    n_event : int
        Number of events
    t_span : 2-member sequence
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. Both t0 and tf must be floats
        or values interpretable by the float conversion function.
    y0 : numpy.ndarray
        Initial (integer) state values
    jac_ode : callable
        Jacobian of the states
    jac_events : callable
        Jacobian of the event rates
    method : string
        scipy.integrate.solve_ivp method
    t_eval : array_like or None, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points selected by the solver.

    See scipy.integrate.solve_ivp for full list of options
    """

    if (method in ['Radau', 'BDF', 'LSODA']) and ( (jac_ode is None) or (jac_events is None) ):
        raise KeyError(f"Need Jacobian for method: {method}")

    stepper = DeterministicSolver(ode_eqns, n_state, event_rates, n_event, jac_ode, jac_events)

    if perf:
        start_time = time.perf_counter()
        start_cpu = time.process_time()

    # result = stepper.integrate(t, y0, t0=t0, **kwargs)

    result = stepper.integrate(
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        dense_output=dense_output,
        events=events,
        vectorized=vectorized,
        args=args,
        **options)

    t = result.t
    y = result.y
    event_counts = result.event_counts
    diag = result.scipy_out
    result = TimeSeriesResult(y=y, t=t, event_counts=event_counts)

    out = Output(result=result, config=DeterministicConfig(), diag=diag)

    if perf:
        wall_time_seconds = time.perf_counter() - start_time
        cpu_time_seconds = time.process_time() - start_cpu

        perf = {}
        perf['wall_time_seconds'] = wall_time_seconds
        perf['cpu_time_seconds'] = cpu_time_seconds

        out.performance = PerformanceMetrics(**perf)

    return out