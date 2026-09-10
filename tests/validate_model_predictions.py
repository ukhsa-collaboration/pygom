from unittest import main, TestCase
import numpy as np

from pygom import SimulateOde, Transition, TransitionType
from tests.test_utils import final_size_sd, final_size_mn

class TestModelEstimate(TestCase):

    def setUp(self):

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
        R0 = 2.0
        gamma = 1.0/4.0
        beta = R0 * gamma
        N = 1_000_000

        model.parameters = [
            ('beta', beta),
            ('gamma', gamma),
            ('N', N)
        ]

        # ----------------
        # Initialise model
        # ----------------
        t0 = 0.0

        I_0 = 50
        x0 = np.array([
            N - I_0,
            I_0,
            0
        ], dtype=np.int64)

        model.initial_values = (x0, t0)

        # ------------
        # Check set up
        # ------------        

        # Ensure that stochastic extinction is sufficiently unlikely
        prob_stochastic_extinction = (1/R0)**I_0
        assert(prob_stochastic_extinction < 1e-6)

        # Ensure that initial infected population is sufficiently small
        assert(I_0/N < 1e-4)

        self.model = model
        self.target_mn = N * final_size_mn(R0)
        self.target_sd = final_size_sd(R0, N)

        self.target_max_time = 1

    def test_SIR(self):
        solution = self.model.solve_stochastic(
            t=1000.0,
            method="fixed_tau",
            tau=0.01,
            iteration=1,
            seed=1,
            checker = "forbidden_state",
            perf = True
        )

        final_size = solution[0].result.y[-1, 2]
        err = abs(final_size - self.target_mn) / self.target_sd

        self.assertTrue(
            err < 10,
            msg="Failed to accurately simulate SIR final size."
        )

        self.assertTrue(
            solution[0].perf.cpu_time_seconds < self.target_max_time,
            msg=f"Took more than {self.target_max_time} to run simulation."
        )

if __name__ == '__main__':
    main()
