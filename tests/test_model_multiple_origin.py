from unittest import main, TestCase

import numpy as np

from pygom import DeterministicOde, SimulateOde, Transition, TransitionType


class TestModelMultipleOrigin(TestCase):

    def setUp(self):
        # Tests the following system, solving the deterministic version
        # A + A -> C
        # A + B -> D
        # \emptyset -> A
        # \emptyset -> B

        self.param_eval = {'k1': 0.001,
                           'k2': 0.01,
                           'k3': 1.2,
                           'k4': 1.0}

        self.x0 = [0, 0, 0, 0]
        self.t = np.linspace(0, 100, 100)

        self.states = ['A', 'B', 'C', 'D']
        self.params = ['k1', 'k2', 'k3', 'k4']

        self.transitions = [
                           Transition(origin=('A', 'A'), destination='C',
                                      equation='A * (A - 1) * k1',
                                      transition_type=TransitionType.T),
                           Transition(origin=('A', 'B'), destination='D',
                                      equation='A * B * k2',
                                      transition_type=TransitionType.T)
                           ]
        # our birth and deaths
        self.birth_deaths = [
                             Transition(origin='A', equation='k3',
                                        transition_type=TransitionType.B),
                             Transition(origin='B', equation='k4',
                                        transition_type=TransitionType.B)
                             ]

    def test_deterministic(self):
        ode = DeterministicOde(self.states,
                               self.params,
                               birth_death=self.birth_deaths,
                               transition=self.transitions)

        ode.parameters = self.param_eval
        ode.initial_values = (self.x0, self.t[0])
        _solution = ode.integrate(self.t[1::])

    def test_stochastic(self):
        ode = SimulateOde(self.states, self.params,
                          birth_death=self.birth_deaths,
                          transition=self.transitions)

        ode.parameters = self.param_eval
        ode.initial_values = (self.x0, self.t[0])
        _simX, _simT = ode.simulate_jump(self.t, 5, parallel=False, full_output=True)

    def tearDown(self):
        self.transitions = None
        self.birth_deaths = None
        self.x0 = None
        self.t = None


if __name__ == '__main__':
    main()
