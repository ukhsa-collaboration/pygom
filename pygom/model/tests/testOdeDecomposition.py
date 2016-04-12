from unittest import TestCase

from pygom import SimulateOdeModel, Transition, TransitionType
import numpy
import sympy

class TestOdeDecomposition(TestCase):

    def test_simple(self):
        ode1 = Transition('S','-beta*S*I', 'ode')
        ode2 = Transition('I','beta*S*I - gamma * I', 'ode')
        ode3 = Transition('R','gamma*I', 'ode')
        stateList = ['S','I','R']
        paramList = ['beta','gamma']
        ode = SimulateOdeModel(stateList, paramList, odeList=[ode1,ode2,ode3])

        ode2 = ode.returnObjWithTransitionsAndBD()
        diffEqZero = map(lambda x: x==0, sympy.simplify(ode.getOde() - ode2.getOde()))

        if numpy.any(numpy.array(diffEqZero) == False):
            raise Exception("Simple: SIR Decomposition failed")

    def test_hard(self):
        stateList = ['S', 'L','I','A','R','D']
        paramList = ['beta','p','kappa','alpha','f','delta','epsilon', 'N']
        odeList = [
            Transition('S', '- beta * S/N * ( I + delta * A)', 'ODE'),
            Transition('L', 'beta * S/N * (I + delta * A) - kappa * L', 'ODE'),
            Transition('I', 'p * kappa * L - alpha * I', 'ODE'),
            Transition('A', '(1-p) * kappa * L - epsilon * A', 'ODE'),
            Transition('R', 'f * alpha * I + epsilon * A', 'ODE'),
            Transition('D', '(1-f) * alpha * I', 'ODE') 
            ]

        ode = SimulateOdeModel(stateList, paramList, odeList=odeList)

        ode2 = ode.returnObjWithTransitionsAndBD()
        diffEqZero = map(lambda x: x==0, sympy.simplify(ode.getOde() - ode2.getOde()))

        if numpy.any(numpy.array(diffEqZero) == False):
            raise Exception("Hard: SLIARD Decomposition failed")


    def test_bd(self):
        stateList = ['S', 'I', 'R']
        paramList = ['beta', 'gamma', 'B', 'mu']
        odeList = [
            Transition(origState='S', 
                       equation='-beta * S * I + B - mu * S',
                       transitionType=TransitionType.ODE),
            Transition(origState='I', 
                       equation='beta * S * I - gamma * I - mu * I',
                       transitionType=TransitionType.ODE),
            Transition(origState='R', 
                       destState='R', 
                       equation='gamma * I',
                       transitionType=TransitionType.ODE)
            ]

        ode = SimulateOdeModel(stateList, paramList, odeList=odeList)

        ode2 = ode.returnObjWithTransitionsAndBD()
        diffEqZero = map(lambda x: x==0, sympy.simplify(ode.getOde() - ode2.getOde()))

        if numpy.any(numpy.array(diffEqZero) == False):
            raise Exception("Birth Death: SIR+BD Decomposition failed")
    

