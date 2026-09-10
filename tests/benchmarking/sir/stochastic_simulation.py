from pygom import SimulateOde, Transition, TransitionType
import numpy as np

stateList = ['S', 'I', 'R']
paramList = ['beta', 'gamma', 'N']

transitionList = [
    Transition(origin='S', destination='I', equation='beta*S*I/N', transition_type=TransitionType.T),
    Transition(origin='I', destination='R', equation='gamma*I', transition_type=TransitionType.T)
]

model = SimulateOde(stateList, paramList, transition=transitionList)

def run_tau_simulation(run_id, seed, params):

    N = params['N']
    gamma = params['gamma']
    beta = params['beta']
    tau = params['tau']
    i0 = params['i0']
    iteration = 1

    model.parameters = [
        ('beta', beta),
        ('gamma', gamma),
        ('N', N)
    ]

    x0 = np.array([N-i0, i0, 0], dtype=np.int64)

    model.initial_values = (x0, 0.0)

    sol = model.solve_stochastic(
        t=1000.0,
        method="fixed_tau",
        tau=tau,
        iteration=iteration,
        perf=True,
        seed=seed
    )

    result = sol[0].result

    y = result.y
    final_size = y[-1, 2]

    peak_idx = np.argmax(y[:, 1])
    peak_I = y[peak_idx, 1]
    time_to_peak_I = result.t[peak_idx]

    return {
        "run_id": run_id,
        "seed": seed,
        "beta": beta,
        "gamma": gamma,
        "N": N,
        "i0": i0,
        "tau": tau,
        "final_size": final_size,
        "cpu_time_seconds":
            sol[0].performance.cpu_time_seconds,
        "wall_time_seconds":
            sol[0].performance.wall_time_seconds,
        "peak_I": peak_I,
        "time_to_peak_I": time_to_peak_I
    }


def run_exact_simulation(run_id, seed, params):

    N = params['N']
    gamma = params['gamma']
    beta = params['beta']
    i0 = params['i0']
    iteration = 1

    model.parameters = [
        ('beta', beta),
        ('gamma', gamma),
        ('N', N)
    ]

    x0 = np.array([N-i0, i0, 0], dtype=np.int64)

    model.initial_values = (x0, 0.0)

    sol = model.solve_stochastic(
        t=1000.0,
        method="direct",
        iteration=iteration,
        perf=True,
        seed=seed
    )

    result = sol[0].result

    y = result.y
    final_size = y[-1, 2]

    peak_idx = np.argmax(y[:, 1])
    peak_I = y[peak_idx, 1]
    time_to_peak_I = result.t[peak_idx]

    return {
        "run_id": run_id,
        "seed": seed,
        "beta": beta,
        "gamma": gamma,
        "N": N,
        "i0": i0,
        "final_size": final_size,
        "cpu_time_seconds":
            sol[0].performance.cpu_time_seconds,
        "wall_time_seconds":
            sol[0].performance.wall_time_seconds,
        "peak_I": peak_I,
        "time_to_peak_I": time_to_peak_I
    }