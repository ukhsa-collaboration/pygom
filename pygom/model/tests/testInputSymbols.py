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
