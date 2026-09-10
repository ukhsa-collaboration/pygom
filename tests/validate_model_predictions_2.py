import numpy as np
import pytest

from pygom import SimulateOde, Transition, TransitionType
from tests.test_utils import final_size_sd, final_size_mn

from dataclasses import dataclass

@dataclass(frozen=True)
class SolverConfig:
    method: str
    checker: str
    kwargs: dict

class GrowthRateBenchmark:
    def __init__(self):
        self.name = "growth_rate"

    def evaluate(self, scenario, solutions):
        observed_r = []
        for solution in solutions:
            inf = scenario.infectious_population(solution)
            t = solution.result.t

            Imin = 1e3
            Imax = 1e4

            mask = (
                (inf >= Imin)
                & (inf <= Imax)
            )

            if np.sum(mask) < 5:
                continue

            coef = np.polyfit(
                t[mask],
                np.log(inf[mask]),
                deg=1
            )

            observed_r.append(coef[0])

        assert len(observed_r) > 0

        mean_observed_r = np.mean(observed_r)

        expected_r = scenario.theoretical_growth_rate()

        assert (
            abs(mean_observed_r - expected_r)
            < scenario.growth_rate_tol
        ), (
            f"Expected growth rate {expected_r:.4f}, "
            f"observed {mean_observed_r:.4f}"
        )

class SIRScenario:

    @dataclass(frozen=True)
    class SIRParameters:
        R0: float
        gamma: float
        N: int

        @property
        def beta(self):
            return self.R0 * self.gamma

    parameter_sets = [
        SIRParameters(
            R0=1.5,
            gamma=0.25,
            N=100_000
        ),
        SIRParameters(
            R0=2.0,
            gamma=0.25,
            N=100_000
        ),
    ]

    def __init__(self):
        self.name = "sir"
        self.iterations = 10
        self.t_end = 1000.0

        self.beta = self.params['beta']
        self.gamma = self.params['gamma']
        self.N = self.params['N']
        self.R0 = self.params['R0']

        self.model = self.build()

        self.benchmarks = [
            GrowthRateBenchmark()
        ]

    def build(self):
        # ---------------
        # Build SIR model
        # ---------------
        stateList = ['S', 'I', 'R']
        paramList = ['beta', 'gamma', 'N']

        transitionList = [
            Transition(origin='S', destination='I', equation='beta*S*I/N', transition_type=TransitionType.T),
            Transition(origin='I', destination='R', equation='gamma*I', transition_type=TransitionType.T)
        ]

        model = SimulateOde(stateList, paramList, transition=transitionList)

        # ------------------
        # Parameterise model
        # ------------------

        model.parameters = [
            ('beta', self.beta),
            ('gamma', self.gamma),
            ('N', self.N)
        ]

        # ----------------
        # Initialise model
        # ----------------
        t0 = 0.0
        I_0 = 50

        x0 = np.array([
            self.N - I_0,
            I_0,
            0
        ], dtype=np.int64)

        model.initial_values = (x0, t0)

        # ------------
        # Check set up
        # ------------        
        # Ensure that stochastic extinction is sufficiently unlikely
        prob_stochastic_extinction = (1/self.R0)**I_0
        assert(prob_stochastic_extinction < 1e-6)

        # Ensure that initial infected population is sufficiently small
        assert(I_0/self.N < 1e-4)

        return model

    def simulate(self, solver):
        solutions = self.model.solve_stochastic(
            t=self.t_end,
            method=solver.method,
            iteration=self.iterations,
            seed=1,
            checker=solver.checker,
            perf=True,
            **solver.kwargs
        )
        return solutions

    def infectious_population(self, solution):
        i = self.model.state_list.index("I")
        return solution.result.y[:, i]

    def theoretical_growth_rate(self):
        return self.beta - self.gamma

    def theoretical_final_size_mn(self):
        return self.N * final_size_mn(self.R0)

    def theoretical_final_size_sd(self):
        return final_size_sd(self.R0, self.N)


SOLVERS = [
    SolverConfig(
        method="fixed_tau",
        checker="forbidden_state",
        kwargs={
            "tau": 0.1,
        }
    )
]

SCENARIOS = [
    SIRScenario(),
]

@pytest.fixture(
    scope="module"
)
def simulation_cache():
    cache = {}
    for scenario in SCENARIOS:
        for solver in SOLVERS:
            key = (
                scenario.name,
                solver.name,
            )
            cache[key] = scenario.simulate(solver)

    return cache


def generate_cases():
    cases = []
    for scenario_cls in SCENARIOS:
        for params in scenario_cls.parameter_sets:
            cases.append(
                scenario_cls(params)
            )

    return cases


@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=lambda s: s.name,
)
@pytest.mark.parametrize(
    "solver",
    SOLVERS,
    ids=lambda s: s.name,
)
def test_growth_rate(
    scenario,
    solver,
    simulation_cache,
):

    solutions = simulation_cache[
        (
            scenario.name,
            solver.name,
        )
    ]

    for benchmark in scenario.benchmarks:

        benchmark.evaluate(
            scenario,
            solutions,
        )
