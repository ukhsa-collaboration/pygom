from unittest import main, TestCase

import numpy as np

from pygom import SimulateOde, Transition, TransitionType


class TestSimulateJump(TestCase):

    def setUp(self):
        n_size = 50
        self.n_sim = 3
        # x0 = [1,1.27e-6,0] # original
        self.x0 = [2362206.0, 3.0, 0.0]
        self.t = np.linspace(0, 250, n_size)
        # use a shorter version if we just want to test
        # whether setting the seed is applicable
        self.t_seed = np.linspace(0, 10, 10)
        self.index = np.random.randint(n_size)

        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma', 'N']
        transition_list = [
                          Transition(origin='S', destination='I',
                                     equation='beta*S*I/N',
                                     transition_type=TransitionType.T),
                          Transition(origin='I', destination='R',
                                     equation='gamma*I',
                                     transition_type=TransitionType.T)
                          ]
        # initialize the model
        self.odeS = SimulateOde(state_list, param_list,
                                transition=transition_list)

        self.odeS.parameters = [0.5, 1.0/3.0, self.x0[0]]
        self.odeS.initial_values = (self.x0, self.t[0])

    def tearDown(self):
        self.odeS = None

    def test_simulate_jump_serial(self):
        """
        Stochastic ode under the interpretation that we have a continuous
        time Markov chain as the underlying process
        """

        solution = self.odeS.integrate(self.t[1::])
        # random evaluation to see if the functions break down
        self.odeS.transition_mean(self.x0, self.t[0])
        self.odeS.transition_var(self.x0, self.t[0])

        self.odeS.transition_mean(solution[self.index,:], self.t[self.index])
        self.odeS.transition_var(solution[self.index,:], self.t[self.index])

        _simX, _simT = self.odeS.simulate_jump(250, self.n_sim, parallel=False, full_output=True)

    def test_simulate_jump_same_seed(self):
        """
        Testing that using the same seed produces the same simulation under
        a CTMC interpretation only under a serial simulation.  When simulating
        with a parallel backend, the result will be different as the seed
        does not propagate through.
        """
        seed = np.random.randint(1000)

        # First note that the default is a parallel simulation using
        # dask as the backend.  This does not use the seed.
        # But if we run it in serial then the seed will be used
        # and the output will be identical
        np.random.seed(seed)
        simX1, simT1 = self.odeS.simulate_jump(self.t_seed[1::], self.n_sim,
                                               parallel=False, full_output=True)
        np.random.seed(seed)
        simX2, simT2 = self.odeS.simulate_jump(self.t_seed[1::], self.n_sim,
                                               parallel=False, full_output=True)

        for i, xi in enumerate(simX1):
            self.assertTrue(np.allclose(simX2[i], xi))

    def test_simulate_jump_different_seed(self):
        """
        Testing that using a different seed produces different simulations
        under a CTMC interpretation regardless of the backend.
        """
        np.random.seed(1)
        simX1, simT1 = self.odeS.simulate_jump(self.t_seed[1::], self.n_sim,
                                               parallel=False, full_output=True)
        np.random.seed(2)
        simX2, simT2 = self.odeS.simulate_jump(self.t_seed[1::], self.n_sim,
                                               parallel=False, full_output=True)

        for i, xi in enumerate(simX1):
            self.assertFalse(np.allclose(simX2[i], xi))


if __name__ == '__main__':
    main()
