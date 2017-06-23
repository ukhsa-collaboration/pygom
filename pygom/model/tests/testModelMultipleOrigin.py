from unittest import TestCase

from pygom import OperateOdeModel, SimulateOdeModel, Transition, TransitionType
import numpy
import sympy
from collections import OrderedDict

## define parameters
paramEval = {'k1':0.001,
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
        transitionList = [
                          Transition(origState=('A','A'), destState='C',
                                     equation='A * (A - 1) * k1',
                                     transitionType=TransitionType.T),
                          Transition(origState=('A','B'), destState='D',
                                     equation='A * B * k2',
                                     transitionType=TransitionType.T)
                          ]
        # our birth and deaths
        birthDeathList = [
                          Transition(origState='A', equation='k3',
                                     transitionType=TransitionType.B),
                          Transition(origState='B', equation='k4',
                                     transitionType=TransitionType.B)
                          ]

        ode = OperateOdeModel(stateList,
                              paramList,
                              birthDeathList=birthDeathList,
                              transitionList=transitionList)

        x0 = [0,0,0,0]
        t = numpy.linspace(0, 100, 100)

        ode.setParameters(paramEval).setInitialValue(x0,t[0])
        solution = ode.integrate(t[1::])
    
    def test_stochastic(self):
        # Tests the following system simulating jumps
        # A + A -> C
        # A + B -> D
        # \emptyset -> A
        # \emptyset -> B
        stateList = ['A', 'B', 'C', 'D']
        paramList = ['k1', 'k2', 'k3', 'k4']
        transitionList = [
                          Transition(origState=('A','A'), destState='C',
                                     equation='A * (A - 1) * k1',
                                     transitionType=TransitionType.T),
                          Transition(origState=('A','B'), destState='D',
                                     equation='A * B * k2',
                                     transitionType=TransitionType.T)
                          ]
        # our birth and deaths
        birthDeathList = [
                          Transition(origState='A', equation='k3',
                                     transitionType=TransitionType.B),
                          Transition(origState='B', equation='k4',
                                     transitionType=TransitionType.B)
                          ]

        ode = SimulateOdeModel(stateList,
                               paramList,
                               birthDeathList=birthDeathList,
                               transitionList=transitionList)

        x0 = [0,0,0,0]
        t = numpy.linspace(0, 100, 100)

        ode.setParameters(paramEval).setInitialValue(x0, t[0])
        simX, simT = ode.simulateJump(t, 5, full_output=True)