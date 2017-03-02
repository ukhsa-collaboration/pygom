from unittest import TestCase

import numpy

from pygom import Transition, TransitionType, OperateOdeModel, ODEVariable

class TestModelVector(TestCase):

    def test_Vector_State1(self):
        # state is a vector
        stateList = ['y1:4']
        paramList = []
        # transitions call from the vector
        transitionList = [
                          Transition(origState='y[0]', destState='y[1]', equation='0.04*y[0]', transitionType=TransitionType.T),
                          Transition(origState='y[1]', destState='y[0]', equation='1e4*y[1]*y[2]', transitionType=TransitionType.T),
                          Transition(origState='y[1]', destState='y[2]', equation='3e7*y[1]*y[1]', transitionType=TransitionType.T)
                          ]
        # initialize the model
        ode = OperateOdeModel(stateList, paramList, transitionList=transitionList)
        ode.getOde()

        t = numpy.append(0, 4*numpy.logspace(-6, 6, 1000))
        ode = ode.setInitialValue([1.0, 0.0, 0.0], t[0])
        # try to integrate to see if there is any problem
        solution, output = ode.integrate(t[1::], full_output=True)
        
    def test_Vector_State2(self):
        # state is a vector
        stateList = ['y1:4']
        paramList = []
        # transitions are explicit names
        transitionList = [
                          Transition(origState='y1', destState='y2', equation='0.04*y1', transitionType=TransitionType.T),
                          Transition(origState='y2', destState='y1', equation='1e4*y2*y3', transitionType=TransitionType.T),
                          Transition(origState='y2', destState='y3', equation='3e7*y2*y2', transitionType=TransitionType.T)
                          ]
 
        ode = OperateOdeModel(stateList, paramList, transitionList=transitionList)
        ode.getOde()

        t = numpy.append(0, 4*numpy.logspace(-6, 6, 1000))
        ode = ode.setInitialValue([1.0, 0.0, 0.0], t[0])
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
                          Transition(origState='y1', destState='y2', equation='0.04*y1', transitionType=TransitionType.T),
                          Transition(origState='y2', destState='y1', equation='1e4*y2*y3', transitionType=TransitionType.T),
                          Transition(origState='y2', destState='y3', equation='3e7*y2*y2', transitionType=TransitionType.T)
                          ]
 
        ode = OperateOdeModel(stateList, paramList, transitionList=transitionList)
        ode.getOde()

        t = numpy.append(0, 4*numpy.logspace(-6, 6, 1000))
        ode = ode.setInitialValue([1.0, 0.0, 0.0], t[0])
        # try to integrate to see if there is any problem
        solution, output = ode.integrate(t[1::], full_output=True)