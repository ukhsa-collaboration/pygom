from unittest import TestCase

from collections import OrderedDict

import numpy
import sympy

from pygom import DeterministicOde, SimulateOde, Transition, TransitionType

## define parameters
param_eval = {'k1':0.001,
              'k2':0.01,
              'k3':1.2,
              'k4':1.0}

class TestModelMultipleOrigin(TestCase):

    def test_deterministic(self):
        # Tests the following system, solving the deterministic version
        # A + A -> C
        # A + B -> D
        # \emptyset -> A
        # \emptyset -> B

        stateList = ['A', 'B', 'C', 'D']
        paramList = ['k1', 'k2', 'k3', 'k4']
        transition_list = [
                           Transition(origin=('A','A'), destination='C',
                                      equation='A * (A - 1) * k1',
                                      transition_type=TransitionType.T),
                           Transition(origin=('A','B'), destination='D',
                                      equation='A * B * k2',
                                      transition_type=TransitionType.T)
                          ]
        # our birth and deaths
        birth_death_list = [
                            Transition(origin='A', equation='k3',
                                       transition_type=TransitionType.B),
                            Transition(origin='B', equation='k4',
                                       transition_type=TransitionType.B)
                           ]

        ode = DeterministicOde(stateList,
                               paramList,
                               birth_death=birth_death_list,
                               transition=transition_list)

        x0 = [0, 0, 0, 0]
        t = numpy.linspace(0, 100, 100)

        ode.parameters = param_eval
        ode.initial_values = (x0, t[0])
        _solution = ode.integrate(t[1::])

    def test_stochastic(self):
        # Tests the following system simulating jumps
        # A + A -> C
        # A + B -> D
        # \emptyset -> A
        # \emptyset -> B
        state = ['A', 'B', 'C', 'D']
        param = ['k1', 'k2', 'k3', 'k4']
        transition_list = [
                          Transition(origin=('A','A'), destination='C',
                                     equation='A * (A - 1) * k1',
                                     transition_type=TransitionType.T),
                          Transition(origin=('A','B'), destination='D',
                                     equation='A * B * k2',
                                     transition_type=TransitionType.T)
                          ]
        # our birth and deaths
        bd_list = [
                   Transition(origin='A', equation='k3',
                              transition_type=TransitionType.B),
                   Transition(origin='B', equation='k4',
                              transition_type=TransitionType.B)
                   ]

        ode = SimulateOde(state, param,
                          birth_death=bd_list,
                          transition=transition_list)

        x0 = [0, 0, 0, 0]
        t = numpy.linspace(0, 100, 100)

        ode.parameters = param_eval
        ode.initial_values = (x0, t[0])
        _simX, _simT = ode.simulate_jump(t, 5, full_output=True)
