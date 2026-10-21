from pygom import SimulateOde, Transition, TransitionType
import numpy as np

from scipy.optimize import minimize_scalar

stateList = ['S', 'I']
paramList = ['beta', 'gamma', 'N']

transitionList = [
    Transition(origin='S', destination='I', equation='beta*S*I/N', transition_type=TransitionType.T),
    Transition(origin='I', destination='S', equation='gamma*I', transition_type=TransitionType.T)
]

model = SimulateOde(stateList, paramList, transition=transitionList)

def run_simulation(run_id, seed, params):

    N = params['N']
    gamma = params['gamma']
    beta = params['beta']
    i0 = params['i0']
    iteration = 1
    R0 = beta / gamma

    model.parameters = [
        ('beta', beta),
        ('gamma', gamma),
        ('N', N)
    ]

    x0 = np.array([N-i0, i0], dtype=np.int64)

    model.initial_values = (x0, 0.0)

    method = params['method']

    if method == 'fixed_tau':
        param = params['tau']

        sol = model.solve_stochastic(
            t=1000.0,
            method=method,
            tau=param,
            iteration=iteration,
            perf=True,
            seed=seed
        )
    elif method == 'cao2006':
        param = params['epsilon']

        sol = model.solve_stochastic(
            t=1000.0,
            method=method,
            epsilon=param,
            iteration=iteration,
            perf=True,
            seed=seed
        )
    elif method == 'ukhsa2026':
        param = params['epsilon']

        sol = model.solve_stochastic(
            t=1000.0,
            method=method,
            epsilon=param,
            iteration=iteration,
            perf=True,
            seed=seed
        )
    elif method == 'direct':
        param = 0

        sol = model.solve_stochastic(
            t=1000.0,
            method=method,
            iteration=iteration,
            perf=True,
            seed=seed
        )

    def get_tau_corr(t, I):
        dt = np.diff(t)

        I_mn = N * (1 - (1 / R0))

        x = I - I_mn

        x0 = x[:-1]
        x1 = x[1:]

        def objective(kappa):
            pred = np.exp(-kappa*dt) * x0
            return np.sum((x1 - pred)**2)

        res = minimize_scalar(
            objective,
            bounds=(0,10),
            method="bounded"
        )

        tau_corr = 1/res.x

        return tau_corr

    result = sol[0].result
    I = result.y[:, 1]
    t = result.t

    # allow burn in of 50% time
    n_t = len(I)
    i_burn = int(0.5 * n_t)
    t = t[i_burn:]
    I = I[i_burn:]

    tau_corr = get_tau_corr(t, I)
    I_mn = I.mean()
    I_sd = I.std()

    out = {
        "run_id": run_id,
        "seed": seed,
        "beta": beta,
        "gamma": gamma,
        "N": N,
        "i0": i0,
        "method": method,
        "param": param,
        "tau_corr": tau_corr,
        "I_mn": I_mn,
        "I_sd": I_sd,
        "cpu_time_seconds":
            sol[0].performance.cpu_time_seconds,
        "wall_time_seconds":
            sol[0].performance.wall_time_seconds
    }

    return out