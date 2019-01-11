from unittest import TestCase

from pygom.model.utils import CompileCanary

class TestCanary(CompileCanary):
    states=['a_state', 'b_state']

class TestModelExisting(TestCase):

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

