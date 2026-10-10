from unittest import main, TestCase

import numpy as np
import scipy.stats

from pygom import SimulateOde, Transition, TransitionType
from pygom.utilR import rgamma
from pygom.model import common_models


class TestSimulateParam(TestCase):

    def setUp(self):
        self.n_sim = 1000
        # initial time
        self.t0 = 0

        i0 = 1.27e-6
        # the initial state, normalized to zero one
        self.x0 = [1-i0, i0, 0]
        # set the time sequence that we would like to observe
        self.t = np.linspace(0, 150, 100)
        # Standard.  Find the solution.
        ode = common_models.SIR_norm()
        ode.parameters = [0.5, 1.0 / 3.0]
        ode.initial_values = (self.x0, self.t0)
        # self.solution = ode.integrate(self.t[1::], full_output=False)

        self.solution = ode.solve_deterministic(t=self.t[1::])[0].result.y

        # now we need to define our ode explicitly
        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma']
        transition_list = [
            Transition(origin='S', destination='I',
                       equation='beta*S*I',
                       transition_type=TransitionType.T),
            Transition(origin='I', destination='R',
                       equation='gamma*I',
                       transition_type=TransitionType.T)
        ]
        # our stochastic version
        self.odeS = SimulateOde(state_list, param_list, transition=transition_list)

    def tearDown(self):
        self.solution = None
        self.odeS = None

    # def test_solve_determ_1(self):
    #     """
    #     Stochastic ode under the interpretation that the parameters follow
    #     some sort of distribution.  In this case, a scipy.distn object.
    #     """

    #     # define our parameters in terms of two gamma distributions
    #     # where the expected values are the same as before [0.5,1.0/3.0]
    #     d = dict()
    #     d['beta'] = scipy.stats.gamma(100.0, 0.0, 1.0/200.0)
    #     d['gamma'] = scipy.stats.gamma(100.0, 0.0, 1.0/300.0)
    #     self.odeS.parameters = d
    #     self.odeS.initial_values = (self.x0, self.t0)

    #     # now we generate the solutions
    #     sim = self.odeS.solve_deterministic(t=self.t[1::], iteration=self.n_sim)[0].result.y

    #     solution_diff = sim - self.solution

    #     # test :)
    #     self.assertTrue(np.any(abs(solution_diff) <= 0.2))

    # def test_solve_determ_2(self):
    #     """
    #     Stochastic ode under the interpretation that the parameters follow
    #     some sort of distribution.  In this case, a function handle which
    #     has the same name as those found in R.
    #     """

    #     # define our parameters in terms of two gamma distributions
    #     # where the expected values are the same as before [0.5,1.0/3.0]
    #     d = dict()
    #     d['beta'] = (rgamma, {'shape': 100.0, 'rate': 200.0})
    #     d['gamma'] = (rgamma, (100.0, 300.0))

    #     self.odeS.parameters = d
    #     self.odeS.initial_values = (self.x0, self.t0)

    #     # now we generate the solutions
    #     sim = self.odeS.solve_determ(self.t[1::], self.n_sim, parallel=False)
    #     solution_diff = sim - self.solution

    #     # test :)
    #     self.assertTrue(np.all(abs(solution_diff) <= 0.2))

    def test_solve_determ_same_seed(self):
        """
        Stochastic ode under the interpretation that the parameters follow
        some sort of distribution and simulating using the same seed
        should produce the same result.
        """

        # # define our parameters in terms of two gamma distributions
        # # where the expected values are the same as before [0.5,1.0/3.0]
        # d = dict()
        # d['beta'] = scipy.stats.gamma(100.0, 0.0, 1.0/200.0)
        # d['gamma'] = scipy.stats.gamma(100.0, 0.0, 1.0/300.0)
        # self.odeS.parameters = d
        # self.odeS.initial_values = (self.x0, self.t0)

        # # now we generate the solutions
        # seed = np.random.randint(1000)
        # np.random.seed(seed)
        # solution1, Yall1 = self.odeS.solve_deterministic(self.t[1::], self.n_sim,
        #                                             parallel=False, full_output=True)
        # np.random.seed(seed)
        # solution2, Yall2 = self.odeS.solve_deterministic(self.t[1::], self.n_sim,
        #                                             parallel=False, full_output=True)

        # self.assertTrue(np.allclose(solution1, solution2))

        # for i, yi in enumerate(Yall1):
        #     self.assertTrue(np.allclose(Yall2[i], yi))

        random_param_set = dict()  # container for random param set
        random_param_set['gamma'] = (rgamma, {'shape':100, 'rate':200})
        random_param_set['beta'] = (rgamma, {'shape':100, 'rate':300})

        self.odeS.parameters = random_param_set
        self.odeS.initial_values = (self.x0, self.t0)

        out1 = self.odeS.solve_deterministic(
            t=self.t,
            iteration=self.n_sim,
            seed=1
        )

        out2 = self.odeS.solve_deterministic(
            t=self.t,
            iteration=self.n_sim,
            seed=1
        )

        for i in range(self.n_sim):
            self.assertTrue(np.allclose(out1[i].result.y, out2[i].result.y))

    def test_solve_determ_same_seed(self):
        """
        Stochastic ode under the interpretation that the parameters follow
        some sort of distribution and simulating using the same seed
        should produce the same result.
        """

        random_param_set = dict()  # container for random param set
        random_param_set['gamma'] = (rgamma, {'shape':100, 'rate':200})
        random_param_set['beta'] = (rgamma, {'shape':100, 'rate':300})

        self.odeS.parameters = random_param_set
        self.odeS.initial_values = (self.x0, self.t0)

        out1 = self.odeS.solve_deterministic(
            t=self.t,
            iteration=self.n_sim,
            seed=1
        )

        out2 = self.odeS.solve_deterministic(
            t=self.t,
            iteration=self.n_sim,
            seed=2
        )

        for i in range(self.n_sim):
            self.assertFalse(np.allclose(out1[i].result.y, out2[i].result.y))

    # def test_solve_determ_different_seed(self):
    #     """
    #     Stochastic ode under the interpretation that the parameters follow
    #     some sort of distribution and simulating using different seeds
    #     should produce different results.
    #     """

    #     # define our parameters in terms of two gamma distributions
    #     # where the expected values are the same as before [0.5,1.0/3.0]
    #     d = dict()
    #     d['beta'] = scipy.stats.gamma(100.0, 0.0, 1.0/200.0)
    #     d['gamma'] = scipy.stats.gamma(100.0, 0.0, 1.0/300.0)
    #     self.odeS.parameters = d
    #     self.odeS.initial_values = (self.x0, self.t0)

    #     # now we generate the solutions
    #     np.random.seed(1)
    #     solution1, Yall1 = self.odeS.solve_deterministic(self.t[1::], 1000,
    #                                                 parallel=False, full_output=True)
    #     np.random.seed(2)
    #     solution2, Yall2 = self.odeS.solve_deterministic(self.t[1::], 1000,
    #                                                 parallel=False, full_output=True)

    #     self.assertFalse(np.allclose(solution1, solution2))

    #     for i, yi in enumerate(Yall1):
    #         self.assertFalse(np.allclose(Yall2, yi))


if __name__ == '__main__':
    main()
