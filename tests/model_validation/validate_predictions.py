import pytest

from dataclasses import dataclass

from models import SIR_Scenario, SEIR_Scenario, SIS_Scenario

#################################################
# Set up suite of compartmental models to study
#################################################
MODELS = [
    SIR_Scenario,
    # SEIR_Scenario,
    SIS_Scenario
]

def generate_scenarios():
    scenarios = []
    for model in MODELS:
        for params in model.parameter_sets:
            scenarios.append(
                model(params)
            )
    return scenarios

SCENARIOS = generate_scenarios()


#################################################
# Set up suite of solver methods
#################################################

@dataclass(frozen=True)
class MethodConfig:
    name: str
    kwargs: dict

@dataclass(frozen=True)
class SolverConfig:
    method: MethodConfig
    checker: str

    @property
    def id(self):
        return f"{self.method.name}-{self.checker}"

METHODS = [
    # MethodConfig(
    #     name="fixed_tau",
    #     kwargs={
    #         "tau": 0.1,
    #     }
    # ),
    MethodConfig(
        name="direct",
        kwargs={}
    ),
    # MethodConfig(
    #     name="ukhsa2026",
    #     kwargs={
    #         "epsilon": 0.1,
    #     }
    # ),
    # MethodConfig(
    #     name="cao2006",
    #     kwargs={
    #         "epsilon": 0.05,
    #     }
    # ),
]

CHECKERS = ["forbidden_state", "forbidden_reaction"]
# REFINERS = ["none", "prob"]

def generate_solvers():
    solvers = []
    for method in METHODS:
        if method.name in ["direct", "first_reaction"]:
            solvers.append(
                SolverConfig(
                    method=method,
                    checker="forbidden_state",
                )
            )
        else:
            for checker in CHECKERS:
                solvers.append(
                    SolverConfig(
                        method=method,
                        checker=checker
                    )
                )
    return solvers

SCENARIOS = generate_scenarios()
SOLVERS = generate_solvers()

#################################################
# Preemptively simulate and cache results to avoid
# unnecessary re-simulation
#################################################

@pytest.fixture(scope="module")
def simulation_cache():
    cache = {}
    for scenario in SCENARIOS:
        for solver in SOLVERS:
            cache[(scenario.id, solver.id)] = scenario.simulate(solver)

    return cache


@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=lambda s: s.id,
)
@pytest.mark.parametrize(
    "solver",
    SOLVERS,
    ids=lambda s: s.id,
)
def test_benchmarks(
    scenario,
    solver,
    simulation_cache):

    solutions = simulation_cache[(scenario.id, solver.id)]

    for benchmark in scenario.benchmarks:
        benchmark.evaluate(scenario, solutions)