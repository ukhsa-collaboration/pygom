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
        ode = common_models.SIR()

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
        ode = common_models.SEIR_Birth_Death_Periodic()
        t = numpy.linspace(0, 100, 1001)
        x0 = [0.0658, 0.0007, 0.0002, 0.]
        ode.initial_values = (x0,0)
        ode.parameters = [0.02, 35.84, 100, 1800, 0.27]
        # try to integrate to see if there is any problem
        _solution, _output = ode.integrate(t[1::], True)


if __name__ == '__main__':
    main()
