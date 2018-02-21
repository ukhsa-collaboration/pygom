from unittest import TestCase

import numpy

from pygom import Transition, TransitionType, DeterministicOde, ODEVariable

class TestModelVector(TestCase):

    def test_Vector_State1(self):
        # state is a vector
        stateList = ['y1:4']
        paramList = []
        # transitions call from the vector
        transitionList = [
                          Transition(origin='y[0]', destination='y[1]', equation='0.04*y[0]', transition_type=TransitionType.T),
                          Transition(origin='y[1]', destination='y[0]', equation='1e4*y[1]*y[2]', transition_type=TransitionType.T),
                          Transition(origin='y[1]', destination='y[2]', equation='3e7*y[1]*y[1]', transition_type=TransitionType.T)
                          ]
        # initialize the model
        ode = DeterministicOde(stateList, paramList, transition=transitionList)
        ode.get_ode_eqn()

        t = numpy.append(0, 4*numpy.logspace(-6, 6, 1000))
        ode.initial_values = ([1.0, 0.0, 0.0], t[0])
        # try to integrate to see if there is any problem
        solution, output = ode.integrate(t[1::], full_output=True)

    def test_Vector_State2(self):
        # state is a vector
        stateList = ['y1:4']
        paramList = []
        # transitions are explicit names
        transitionList = [
                          Transition(origin='y1', destination='y2', equation='0.04*y1', transition_type=TransitionType.T),
                          Transition(origin='y2', destination='y1', equation='1e4*y2*y3', transition_type=TransitionType.T),
                          Transition(origin='y2', destination='y3', equation='3e7*y2*y2', transition_type=TransitionType.T)
                          ]

        ode = DeterministicOde(stateList, paramList, transition=transitionList)
        ode.get_ode_eqn()

        t = numpy.append(0, 4*numpy.logspace(-6, 6, 1000))
        ode.initial_values = ([1.0, 0.0, 0.0], t[0])
        # try to integrate to see if there is any problem
        solution, output = ode.integrate(t[1::], full_output=True)

    def test_Vector_State3(self):
        # state is a vector
        stateList = [ODEVariable('y1', 'y1'),
                     ODEVariable('y2', 's'),
                     ODEVariable('y3', 'x')]
        paramList = []
        # transitions are explicit names
        transitionList = [
                          Transition(origin='y1', destination='y2', equation='0.04*y1', transition_type=TransitionType.T),
                          Transition(origin='y2', destination='y1', equation='1e4*y2*y3', transition_type=TransitionType.T),
                          Transition(origin='y2', destination='y3', equation='3e7*y2*y2', transition_type=TransitionType.T)
                          ]

        ode = DeterministicOde(stateList, paramList, transition=transitionList)
        ode.get_ode_eqn()

        t = numpy.append(0, 4*numpy.logspace(-6, 6, 1000))
        ode.initial_values = ([1.0, 0.0, 0.0], t[0])
        # try to integrate to see if there is any problem
        solution, output = ode.integrate(t[1::], full_output=True)