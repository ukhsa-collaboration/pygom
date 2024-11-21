from unittest import main, TestCase

import numpy
import scipy.integrate

from pygom.model import common_models


class TestModelExisting(TestCase):

    def test_SIR(self):
        '''
        Test the SIR model from the set of pre-defined models in common_models
        '''
        # We we wish to test another (simpler) model
        ode = common_models.SIR_norm()

        # define the parameters
        param_eval = [
                      ('beta', 0.5),
                      ('gamma', 1.0/3.0)
                     ]

        ode.parameters = param_eval
        # the initial state, normalized to zero one
        initial_state = [1, 1.27e-6, 0]

        # evaluating the ode
        ode.ode(initial_state, 1)
        ode.jacobian(initial_state, 1)
        ode.grad(initial_state, 1)
        # b.sensitivity(sensitivity, t, state)
        ode.sensitivity(numpy.zeros(6), 1, initial_state)

        ode.linear_ode()
        # set the time sequence that we would like to observe
        t = numpy.linspace(1, 150, 100)
        # now find the solution
        _solution, output = scipy.integrate.odeint(ode.ode,
                                                   initial_state,
                                                   t,
                                                   full_output=True)

        self.assertTrue(output['message'] == 'Integration successful.')
        # Happy! :)

    def test_SEIR_periodic(self):
        '''
        Test the SEIR model from the set of pre-defined models in common_models
        '''

        # Set up PyGOM object
        alpha=1/2
        gamma=1/4
        beta0=0.3
        delta=0.2
        period=365

        ode = common_models.SEIR_Birth_Death_Periodic({'alpha':alpha,
                                                       'gamma':gamma,
                                                       'beta0':beta0,
                                                       'delta':delta,
                                                       'period':period})

        N=1e4
        E0=10
        I0=0
        R0=0
        S0=N-(E0+I0+R0)
        
        x0 = [S0, E0, I0, R0, N]
        ode.initial_values = (x0, 0)

        t = numpy.linspace(0, 100, 1001)
        # try to integrate to see if there is any problem
        _solution, _output = ode.integrate(t[1::], True)


if __name__ == '__main__':
    main()
