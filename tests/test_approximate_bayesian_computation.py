#!/usr/bin/env python
# coding: utf-8

from unittest import main, TestCase

import numpy as np

from pygom import SquareLoss, NormalLoss
from pygom.model import common_models
import pygom.approximate_bayesian_computation as pgabc


class TestABC(TestCase):

    def setUp(self):
        # inital values 
        self.x0 = [-1.0, 1.0]

        # params
        self.param_eval = [('a', 0.2), ('b', 0.2),('c', 3.0)]
        # the time points for our observations
        self.t = np.linspace(0, 20, 30).astype('float64')
        self.ode = common_models.FitzHugh(self.param_eval)
        self.ode.initial_values = (self.x0, self.t[0])

        # Standard.  Find the solution which we will be used as
        # "observations later"
        self.solution = self.ode.integrate(self.t[1::])

        # create a noisy set from the clean solution
        self.noised = self.solution + np.random.random(self.solution.shape) * 0.1 * self.solution.max()

    def test_abc_fit(self):
        # fit to the noisy data using ABC and see if we get a similar estimate
        # of parameter to what we gave
        
        # parameters for the ABC fit
        num_samples = 500
        generations = 20
        max_delta = 0.5

        # setup the fit model
        fit_model = common_models.FitzHugh()
        fit_model.parameters = {'a': 0.2, 'b': 0.2, 'c': 3.0}
        fit_model.initial_values = (self.x0, self.t[0])

        # setting up the abc
        # initial guess isn't used by abc so specify anything
        initial_guess = [1] * 3
        
        # set the same prior distribution for each parameter
        boxBounds = [(0,5)] * 3
        
        #setup the loss object 
        modelobj = SquareLoss(theta=initial_guess,
                              ode=fit_model, 
                              x0=self.x0, 
                              t0=self.t[0],
                              t=self.t[1::],
                              y=self.noised[1::,],
                              state_name=['V', 'R'], 
                              target_param=['a', 'b', 'c']
                             )

        modelabc = pgabc.ABC(modelobj, boxBounds, log10scale=[0]*3)

        modelabc.get_posterior_sample(N=num_samples, 
                                      tol=np.inf, 
                                      G=generations, 
                                      q=0.5, 
                                      progress=False)

        # we could plot the result
        # modelabc.plot_pointwise_predictions()
        # modelabc.plot_posterior_histograms()

        # Test that the results are almost equal to the values given (within 20%)
        for res, given in zip(np.median(modelabc.res, axis=0), 
                              np.array([x[1] for x in self.param_eval])):
            self.assertAlmostEqual(res, given, delta=given * max_delta)
