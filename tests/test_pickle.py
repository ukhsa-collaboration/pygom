import pickle
import io
import numpy

from unittest import TestCase
from pygom.model import Transition, TransitionType, SimulateOde

class TestPickling(TestCase):
    
    def setUp(self):
        stateList = ['a', 'x', 'y', 'b']
        paramList = ['k0', 'k1', 'k2']
        transitionList = [
                    Transition(origin='a', 
                               destination='x',
                               equation='k0*a*x', 
                               transition_type=TransitionType.T),
                    Transition(origin='x', 
                               destination='y', 
                               equation='k1*x*y', 
                               transition_type=TransitionType.T),
                    Transition(origin='y', 
                               destination='b', 
                               equation='k2*y', 
                               transition_type=TransitionType.T)
                    ]
        self.ode = SimulateOde(stateList, paramList, transition=transitionList)

    def test_pickle(self):
        '''
        Can we pickle and unpickle an ode object?
        '''
        #cause some compilation to happen
        x0 = [150.0, 10.0, 10.0, 0.0]
        t = numpy.linspace(0, 15, 100)
        self.ode.initial_values = (x0, t[0])
        self.ode.parameters = [0.01, 0.1, 1.0]
        
        self.ode.ode(x0,0)
        with io.BytesIO() as mem_stream:
            pickle.dump(self.ode, mem_stream)
            mem_stream.seek(0)
            ode2 = pickle.load(mem_stream)
            
        self.assertEqual(self.ode,
                         ode2,
                         'Pickled and unpickled objects must be equal')
