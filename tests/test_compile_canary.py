from unittest import TestCase

from copy import deepcopy

from pygom.model.ode_utils import CompileCanary

class TestCanary(CompileCanary):
    states=['a_state', 'b_state']

class TestCompileCanary(TestCase):

    def test_canary(self):
        '''
        Test the CompileCanary class
        '''
        #create the test object
        test_object = TestCanary()

        #Are all the states true
        self.assertTrue(all([test_object.a_state,
                             test_object.b_state]),
                        'All states must start tripped (True)')

        #accessing a non-existent state raises an error
        with self.assertRaises(AttributeError):
            test_object.c_state

        #Reseting a state resets that state
        test_object.a_state = False
        self.assertFalse(test_object.a_state,
                        'Reseting a state must work via assignment')
        #and has no effect on the other state
        self.assertTrue(test_object.b_state,
                        'Resetting a state must not alter other states')

        #does reset work
        test_object.trip()
        self.assertTrue(all([test_object.a_state,
                             test_object.b_state]),
                        'All states must be reset when tripped (True)')
        #reset a state via the reset method
        test_object.reset('b_state')
        self.assertFalse(test_object.b_state,
                        'Reseting a state must work via reset method')
        #and has no effect on the other state
        self.assertTrue(test_object.a_state,
                        'Resetting a state must not alter other states')

    def test_deepcopy(self):
        '''
        Test that CompileCanary can be deep copied without error
        '''
        test_object = CompileCanary()
        test_object.states.extend(['S', 'E', 'I'])
        test_object.trip()
        copied_object = deepcopy(test_object)

        self.assertDictEqual(test_object._states,
                             copied_object._states,
                             'CompileCanary must be capable of being deepcopy(ed)')
