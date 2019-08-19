'''
Created on 14 Jan 2019

@author: thomas.finnie
'''
class CompileCanary(object):
    '''
    Hold the need for (re-)compilation for various functions

    A subclass of this should specify the states to watch

    They may all be tripped to True using the trip() method
    An individual may be reset with the reset() method or with
    a direct assignment (they may not be tripped in this way).
    '''
    states = []

    def __init__(self):
        '''
        Inits the class. Sets up the canaries
        '''
        #set up the states
        self.trip()

    def trip(self):
        '''
        Trip all the canaries
        Returns
        -------
        None
        '''
        self._states = dict([(state, True) for state in self.states])

    def reset(self, name):
        '''
        Reset a canary

        Parameters
        ----------
        name: string
            the name of the canary to reset

        Returns
        -------
        None
        '''
        self.__setattr__(name, False)

    def __deepcopy__(self, memo):
        copied = CompileCanary()
        copied.states = list(self._states.keys())
        copied.trip()
        return copied

    def __getattr__(self, name):
        '''
        Implement a method to get the values of attributes
        '''
        if name == '_states':
            return self._states

        if name in self._states:
            return self._states[name]
        msg = "'{0}' object has no attribute '{1}'"
        raise AttributeError(msg.format(type(self).__name__, name))

    def __setattr__(self, name, value):
        '''
        Check attribute assignment
        '''
        if name in self.states:
            if not value:
                self._states[name] = False
        else:
            object.__setattr__(self, name, value)
