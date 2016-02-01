from unittest import TestCase

import numpy
import scipy.integrate

from pygom import common_models

class TestModelExisting(TestCase):
    def test_SIR(self):
        '''
        Test the SIR model from the set of pre-defined models in common_models
        '''
        # We we wish to test another (simpler) model
        ode = common_models.SIR()

        # define the parameters
        paramEval = [
                     ('beta',0.5), 
                     ('gamma',1.0/3.0)
                     ]

        ode.setParameters(paramEval)
        # the initial state, normalized to zero one
        initialState = [1,1.27e-6,0]
    
        # evaluating the ode
        ode.ode(initialState,1)
        ode.Jacobian(initialState,1)
        ode.Grad(initialState,1)
        # b.sensitivity(sensitivity, t, state)
        ode.sensitivity(numpy.zeros(6), 1, initialState)
    
        ode.isOdeLinear()
        # set the time sequence that we would like to observe
        t = numpy.linspace(1, 150, 100)
        # now find the solution
        soltion,output = scipy.integrate.odeint(ode.ode,initialState,t,full_output=True)
        if output['message']!='Integration successful.':
            raise Exception("Failed integration")
        
        # Happy! :)
        
    def test_SEIR_periodic(self):
        '''
        Test the SEIR model from the set of pre-defined models in common_models
        '''
        ode = common_models.SEIR_Birth_Death_Periodic()
        t = numpy.linspace(0,100,1001)
        x0 = [0.0658,0.0007,0.0002,0.]
        ode.setInitialValue(x0,0).setParameters([0.02,35.84,100,1800,0.27])
        # try to integrate to see if there is any problem
        solution,output=ode.integrate(t[1::],True)