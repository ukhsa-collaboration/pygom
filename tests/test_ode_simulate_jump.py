from unittest import main, TestCase

import numpy as np

from pygom import SimulateOde, Transition, TransitionType


class TestSimulateJump(TestCase):

    def setUp(self):
        n_size = 50
        self.n_sim = 3
        # x0 = [1,1.27e-6,0] # original

        N = 10_000
        i0 = 50

        self.x0 = np.array([N-i0, i0, 0], dtype=np.int64)
        self.t = np.linspace(0, 100, n_size)

        # use a shorter version if we just want to test
        # whether setting the seed is applicable
        self.t_seed = np.linspace(0, 10, 10)
        self.index = np.random.randint(n_size)

        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma', 'N']
        transition_list = [
            Transition(origin='S', destination='I', equation='beta*S*I/N', transition_type=TransitionType.T),
            Transition(origin='I', destination='R', equation='gamma*I', transition_type=TransitionType.T)]
        
        # initialize the model
        self.odeS = SimulateOde(state_list, param_list, transition=transition_list)

        self.odeS.parameters = {"beta": 0.5, "gamma": 1.0/3.0, "N": N}
        self.odeS.initial_values = (self.x0, self.t[0])

    def tearDown(self):
        self.odeS = None

    # def test_simulate_jump_serial(self):
    #     """
    #     Stochastic ode under the interpretation that we have a continuous
    #     time Markov chain as the underlying process
    #     """

    #     solution = self.odeS.integrate(self.t[1::])
    #     # random evaluation to see if the functions break down
    #     self.odeS.transition_mean(self.x0, self.t[0])
    #     self.odeS.transition_variance(self.x0, self.t[0])

    #     self.odeS.transition_mean(solution[self.index,:], self.t[self.index])
    #     self.odeS.transition_variance(solution[self.index,:], self.t[self.index])

    #     result1 = self.odeS.solve_stochastic(
    #         t=250.0,
    #         method="fixed_tau",
    #         tau=0.1,
    #         iteration=self.n_sim,
    #         parallel=False,
    #         seed=0)

    def test_simulate_jump_same_seed(self):
        """
        Testing that using the same seed produces the same simulation under
        a CTMC interpretation only under a serial simulation.  When simulating
        with a parallel backend, the result will be different as the seed
        does not propagate through.
        """
        # seed = np.random.randint(1000)
        # First note that the default is a parallel simulation using
        # dask as the backend.  This does not use the seed.
        # But if we run it in serial then the seed will be used
        # and the output will be identical
        # np.random.seed(seed)

        result1 = self.odeS.solve_stochastic(
            t=self.t[1::],
            method="fixed_tau",
            tau=0.1,
            iteration=self.n_sim,
            parallel=False,
            seed=1)

        result2 = self.odeS.solve_stochastic(
            t=self.t[1::],
            method="fixed_tau",
            tau=0.1,
            iteration=self.n_sim,
            parallel=False,
            seed=1)

        for i in range(self.n_sim):
            self.assertTrue(np.allclose(result1[i].result.y, result2[i].result.y))

    def test_simulate_jump_different_seed(self):
        """
        Testing that using a different seed produces different simulations
        under a CTMC interpretation regardless of the backend.
        """

        result1 = self.odeS.solve_stochastic(
            t=self.t[1::],
            method="fixed_tau",
            tau=0.1,
            iteration=self.n_sim,
            parallel=False,
            seed=1)

        result2 = self.odeS.solve_stochastic(
            t=self.t[1::],
            method="fixed_tau",
            tau=0.1,
            iteration=self.n_sim,
            parallel=False,
            seed=2)

        for i in range(self.n_sim):
            self.assertFalse(np.allclose(result1[i].result.y, result2[i].result.y))




if __name__ == '__main__':
    main()
