from unittest import TestCase

from pygom import Transition, TransitionType, SimulateOde, ODEVariable

class TestParameterActions(TestCase):

    def test_adding_parameters(self):
        '''
        Test adding parameters to a model
        '''
        expected_result =[ODEVariable('beta', 'beta', None, True),
                          ODEVariable('gamma', 'gamma', None, True),
                          ODEVariable('mu', 'mu', None, True),
                          ODEVariable('B', 'B', None, True)
                          ]
        # Model parts
        stateList = ['S', 'I', 'R']
        paramList = ['beta', 'gamma']
        
        # build the basic model
        t1 = Transition(origin='S', 
                        destination='I', 
                        equation='beta*S*I', 
                        transition_type=TransitionType.T)

        t2 = Transition(origin='I', 
                        destination='R', 
                        equation='gamma*I',
                         transition_type=TransitionType.T)

        modelTrans = SimulateOde(stateList,
                                 paramList,
                                 transition=[t1, t2]
                                 )
        
        # add to the parameters
        modelTrans.param_list = paramList + ['mu', 'B']
        
        self.assertListEqual(modelTrans.param_list, 
                             expected_result, 
                             'Adding parameters does not give expected '
                             'parameter list')