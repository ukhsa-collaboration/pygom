import unittest
import numpy as np

from pygom import Transition, TransitionType, SimulateOde, ODEVariable


class TestModelVector(unittest.TestCase):
    @unittest.skip("In adding state limits, this functionality may be lost")
    def test_Vector_State1(self):
        # state is a vector
        state_list = ['y1:4']
        param_list = []
        # transitions call from the vector
        transition_list = [
            Transition(
                origin='y[0]',
                destination='y[1]',
                equation='0.04*y[0]',
                transition_type=TransitionType.T),
            Transition(
                origin='y[1]',
                destination='y[0]',
                equation='1e4*y[1]*y[2]',
                transition_type=TransitionType.T),
            Transition(
                origin='y[1]',
                destination='y[2]',
                equation='3e7*y[1]*y[1]',
                transition_type=TransitionType.T)
        ]
        # initialize the model
        ode = SimulateOde(state_list, param_list, transition=transition_list)
        ode.ode.get_equation()

        t = np.append(0, 4*np.logspace(-6, 6, 1000))
        ode.initial_values = ([1.0, 0.0, 0.0], t[0])
        # try to integrate to see if there is any problem
        _solution, _output = ode.integrate(t[1::], full_output=True)

    def test_Vector_State2(self):
        # state is a vector
        state_list = ['y1:4']
        param_list = []
        # transitions are explicit names
        transition_list = [
            Transition(
                origin='y1',
                destination='y2',
                equation='0.04*y1',
                transition_type=TransitionType.T
            ),
            Transition(
                origin='y2',
                destination='y1',
                equation='1e4*y2*y3',
                transition_type=TransitionType.T),
            Transition(
                origin='y2',
                destination='y3',
                equation='3e7*y2*y2',
                transition_type=TransitionType.T)
            ]

        model = SimulateOde(state_list, param_list, transition=transition_list)
        model.ode.get_equation()

        t = np.append(0, 4*np.logspace(-6, 6, 1000))
        model.initial_values = ([1.0, 0.0, 0.0], t[0])
        # try to integrate to see if there is any problem
        out = model.solve_deterministic(t[1::], full_output=True)

    def test_Vector_State3(self):
        # state is a vector
        state_list = [
            ODEVariable('y1'),
            ODEVariable('y2'),
            ODEVariable('y3')
        ]
        param_list = []
        # transitions are explicit names
        transition_list = [
            Transition(
                origin='y1',
                destination='y2',
                equation='0.04*y1',
                transition_type=TransitionType.T
            ),
            Transition(
                origin='y2',
                destination='y1',
                equation='1e4*y2*y3',
                transition_type=TransitionType.T),
            Transition(
                origin='y2',
                destination='y3',
                equation='3e7*y2*y2',
                transition_type=TransitionType.T)
            ]

        model = SimulateOde(state_list, param_list, transition=transition_list)
        model.ode.get_equation()

        t = np.append(0, 4*np.logspace(-6, 6, 1000))
        model.initial_values = ([1.0, 0.0, 0.0], t[0])
        # try to integrate to see if there is any problem
        out = model.solve_deterministic(t[1::], full_output=True)


if __name__ == '__main__':
    unittest.main()
