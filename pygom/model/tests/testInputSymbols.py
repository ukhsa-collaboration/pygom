from unittest import TestCase

from pygom.model.transition import TransitionType, Transition
from pygom.model.deterministic import OperateOdeModel

class TestInputSymbols(TestCase):
    def test_Signs(self):
        '''
        Making sure that the illegal symbols are catched
        '''
        stateList = [['S+'], ['S-'], ['S*'], ['S\\'], ['_S']]
        paramList = ['beta']
        ode = OperateOdeModel(['S'], paramList)
    
        totalFail = 0
        for state in stateList:
            try:
                ode = OperateOdeModel(state, paramList)
            except:
                totalFail += 1
            
        if totalFail != len(stateList):
            raise Exception("We passed some of the illegal input...")
        # Happy! :)

    def test_multi_symbol_in_str(self):
        stateList = 'S L I A R D'
        paramList = 'beta,p, kappa,  alpha, f delta  epsilon,N'
        odeList = [
            Transition('S', '- beta * S/N * ( I + delta * A)', 'ODE'),
            Transition('L', 'beta * S/N * (I + delta * A) - kappa * L', 'ODE'),
            Transition(origState='I', equation='p * kappa * L - alpha * I', transitionType='ODE'),
            Transition(origState='A', equation='(1-p) * kappa * L - epsilon * A', transitionType='ODE'),
            Transition(origState='R', equation='f * alpha * I + epsilon * A', transitionType='ODE'),
            Transition(origState='D', equation='(1-f) * alpha * I', transitionType='ODE') 
            ]

        ode = OperateOdeModel(stateList, paramList, odeList=odeList)
        # this should not throw an error if the model is initialized correctly
        A = ode.getOde()
