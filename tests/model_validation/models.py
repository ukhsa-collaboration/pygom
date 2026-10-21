import numpy as np

from pygom import SimulateOde, Transition, TransitionType
from test_utils import final_size_sd, final_size_mn

from benchmarks import (
    GrowthRateBenchmark,
    FinalSizeBenchmark,
    EndemicMeanBenchmark,
    EndemicSDBenchmark
)

from dataclasses import dataclass

# Global params
TOL = 1e-1
POP = 10_000
SEED = 123

###################
# SIR 
###################

class SIR_Scenario:

    @dataclass(frozen=True)
    class SIRParameters:
        R0: float
        gamma: float
        N: int

        @property
        def beta(self) -> float:
            return self.R0 * self.gamma

    parameter_sets = [
        SIRParameters(
            R0=1.5,
            gamma=1/4,
            N=POP
        ),
        SIRParameters(
            R0=3.0,
            gamma=1/4,
            N=POP
        )
    ]

    def __init__(self, params):
        self.name = "sir"
        self.iterations = 10
        self.t_end = 1000.0

        self.beta = params.beta
        self.gamma = params.gamma
        self.N = params.N
        self.R0 = params.R0

        self.model = self.build()

        self.benchmarks = [
            GrowthRateBenchmark(),
            FinalSizeBenchmark()
        ]

        self.growth_rate_tol = TOL
        self.final_size_tol = TOL

    @property
    def id(self):
        return (
            f"{self.name}_R0_{self.R0:.1f}"
        )

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
        I_0 = 10

        x0 = np.array([
            self.N - I_0,
            I_0,
            0
        ], dtype=np.int64)

        model.initial_values = (x0, t0)

        # ------------
        # Check set up
        # ------------        
        # Ensure that prob of all simulations going stochastically extinct
        # is sufficiently unlikely
        prob_stochastic_extinction = ((1/self.R0)**I_0)**self.iterations
        assert(prob_stochastic_extinction < 1e-6)

        # Ensure that initial infected population is sufficiently small
        assert(I_0/self.N < 1e-2)

        return model

    def simulate(self, solver):
        solutions = self.model.solve_stochastic(
            t=self.t_end,
            method=solver.method.name,
            iteration=self.iterations,
            seed=SEED,
            checker=solver.checker,
            # refiner=solver.refiner,
            perf=True,
            **solver.method.kwargs
        )
        return solutions

    def infectious_population(self, solution):
        i = self.model.get_state_index("I")
        return solution.result.y[:, i]

    def cumulative_infected_population(self, solution):
        # I know this isn't precisely true but ....
        i = self.model.get_state_index("R")
        return solution.result.y[:, i]

    def theoretical_growth_rate(self):
        return self.beta - self.gamma

    def theoretical_final_size_mn(self):
        return self.N * final_size_mn(self.R0)

    def theoretical_final_size_sd(self):
        return final_size_sd(self.R0, self.N)


###################
# SEIR 
###################

class SEIR_Scenario:

    @dataclass(frozen=True)
    class SIRParameters:
        R0: float
        alpha: float
        gamma: float
        N: int

        @property
        def beta(self) -> float:
            return self.R0 * self.gamma

    parameter_sets = [
        SIRParameters(
            R0=1.5,
            alpha=1/2,
            gamma=1/4,
            N=POP
        ),
        SIRParameters(
            R0=3.0,
            alpha=1/2,
            gamma=1/4,
            N=POP
        )
    ]

    def __init__(self, params):
        self.name = "seir"
        self.iterations = 10
        self.t_end = 1000.0

        self.beta = params.beta
        self.alpha = params.alpha
        self.gamma = params.gamma
        self.N = params.N
        self.R0 = self.beta / self.gamma

        self.model = self.build()

        self.benchmarks = [
            GrowthRateBenchmark(),
            FinalSizeBenchmark()
        ]

        self.growth_rate_tol = TOL
        self.final_size_tol = TOL

    @property
    def id(self):
        return (
            f"{self.name}_R0_{self.R0:.1f}"
        )

    def build(self):
        # ---------------
        # Build SIR model
        # ---------------
        stateList = ['S', 'E', 'I', 'R']
        paramList = ['beta', 'alpha', 'gamma', 'N']

        transitionList = [
            Transition(origin='S', destination='E', equation='beta*S*I/N', transition_type=TransitionType.T),
            Transition(origin='E', destination='I', equation='alpha*E', transition_type=TransitionType.T),
            Transition(origin='I', destination='R', equation='gamma*I', transition_type=TransitionType.T)
        ]

        model = SimulateOde(stateList, paramList, transition=transitionList)

        # ------------------
        # Parameterise model
        # ------------------

        model.parameters = [
            ('beta', self.beta),
            ('alpha', self.alpha),
            ('gamma', self.gamma),
            ('N', self.N)
        ]

        # ----------------
        # Initialise model
        # ----------------
        t0 = 0.0
        E_0 = 10

        x0 = np.array([
            self.N - E_0,
            E_0,
            0,
            0
        ], dtype=np.int64)

        model.initial_values = (x0, t0)

        # ------------
        # Check set up
        # ------------        
        # Ensure that prob of all simulations going stochastically extinct
        # is sufficiently unlikely
        prob_stochastic_extinction = ((1/self.R0)**E_0)**self.iterations
        assert(prob_stochastic_extinction < 1e-6)

        # Ensure that initial infected population is sufficiently small
        assert(E_0/self.N < 1e-2)

        return model

    def simulate(self, solver):
        solutions = self.model.solve_stochastic(
            t=self.t_end,
            method=solver.method.name,
            iteration=self.iterations,
            seed=SEED,
            checker=solver.checker,
            # refiner=solver.refiner,
            perf=True,
            **solver.method.kwargs
        )
        return solutions

    def infectious_population(self, solution):
        i = self.model.get_state_index("I")
        return solution.result.y[:, i]

    def cumulative_infected_population(self, solution):
        # I know this isn't precisely true but ....
        i = self.model.get_state_index("R")
        return solution.result.y[:, i]

    def theoretical_growth_rate(self):
        alpha = self.alpha
        gamma = self.gamma
        beta = self.beta

        return (
            -(alpha + gamma)
            + np.sqrt((alpha - gamma)**2 + 4 * alpha * beta)
        ) / 2

    def theoretical_final_size_mn(self):
        return self.N * final_size_mn(self.R0)

    def theoretical_final_size_sd(self):
        return final_size_sd(self.R0, self.N)


###################
# SIS 
###################

class SIS_Scenario:

    @dataclass(frozen=True)
    class SISParameters:
        R0: float
        gamma: float
        N: int

        @property
        def beta(self) -> float:
            return self.R0 * self.gamma

    parameter_sets = [
        SISParameters(
            R0=1.05,
            gamma=1/50,
            N=POP
        ),
        # SISParameters(
        #     R0=1.1,
        #     gamma=1/50,
        #     N=POP
        # )
    ]

    def __init__(self, params):
        self.name = "sis"
        self.iterations = 10
        self.t_end = 1000.0

        self.beta = params.beta
        self.gamma = params.gamma
        self.N = params.N
        self.R0 = self.beta / self.gamma

        self.model = self.build()

        self.benchmarks = [
            EndemicMeanBenchmark(),
            EndemicSDBenchmark()
        ]

        self.endemic_mn_tol = TOL
        self.endemic_sd_tol = TOL

    @property
    def id(self):
        return (
            f"{self.name}_R0_{self.R0:.1f}"
        )

    def build(self):
        # ---------------
        # Build SIR model
        # ---------------
        stateList = ['S', 'I']
        paramList = ['beta', 'gamma', 'N']

        transitionList = [
            Transition(origin='S', destination='I', equation='beta*S*I/N', transition_type=TransitionType.T),
            Transition(origin='I', destination='S', equation='gamma*I', transition_type=TransitionType.T)
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

        # Release from close to endemic equilibrium:
        S_0 = int(0.9 * ((1 / self.R0) * self.N))

        x0 = np.array([
            S_0,
            self.N - S_0,
        ], dtype=np.int64)

        model.initial_values = (x0, t0)

        return model

    def simulate(self, solver):
        solutions = self.model.solve_stochastic(
            t=self.t_end,
            method=solver.method.name,
            iteration=self.iterations,
            seed=SEED,
            checker=solver.checker,
            # refiner=solver.refiner,
            perf=True,
            **solver.method.kwargs
        )
        return solutions

    def infectious_population(self, solution):
        i = self.model.get_state_index("I")
        return solution.result.y[:, i]

    def theoretical_endemic_mean(self):
        return self.N * (1 - (1/self.R0))

    def theoretical_endemic_sd(self):
        return np.sqrt(self.gamma * self.N * (1 - (1/self.R0)) / (self.beta - self.gamma))

    def corr_time(self):
        return 1.0 / (self.beta - self.gamma)