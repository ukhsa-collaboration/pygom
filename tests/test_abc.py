import numpy as np
from unittest import main, TestCase

from pygom import SquareLoss, NormalLoss
from pygom.model import common_models
from pygom import approximate_bayesian_computation as pgabc


class TestABC(TestCase):
    
    def setUp(self):
        # define the model and parameters
        self.ode = common_models.SIR({'beta':0.5, 'gamma':1.0/3.0})
        
        # the initial state, normalized to one
        self.x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        self.t = np.linspace(0, 150, 100)
        self.ode.initial_values = (self.x0, self.t[0])
        # find the solution
        self.solution = self.ode.integrate(self.t[1::])
        
        # what the posterior median estimates should be close to
        self.target = np.array([0.5, 1.0/3.0])
        
        
    def test_SIR_abc_SquareLoss(self):
        y = self.solution[1::, 1:3]
        
        # setting the parameters in the inference
        parameters = [pgabc.Parameter('beta', 'unif', 0, 3, logscale=False),
                      pgabc.Parameter('gamma', 'unif', 0, 3, logscale=False)]
        
        # creating the loss and abc objects
        sir_obj = pgabc.create_loss(SquareLoss, parameters, self.ode, self.x0, self.t[0],
                                  self.t[1::], y, ['I', 'R'])
        sir_abc = pgabc.ABC(sir_obj, parameters)
        
        # getting the posterior sample
        sir_abc.get_posterior_sample(N=100, tol=np.inf, G=10, q=0.5)
        sir_abc.continue_posterior_sample(N=100, tol=sir_abc.next_tol, G=10, q=0.5)

        # the estimate for beta must be between 0.485 and 0.515
        # the estimate for gamma must be between 0.32 and 0.3466        
        med_est = np.median(sir_abc.res, axis=0)
        self.assertTrue(np.allclose(med_est, self.target, 1e-2, 1e-2))
        
        
    def test_SIR_abc_SquareLoss_MNN(self):
        y = self.solution[1::, 1:3]
        parameters = [pgabc.Parameter('beta', 'unif', 0, 3, logscale=False),
                      pgabc.Parameter('gamma', 'unif', 0, 3, logscale=False)]
        sir_obj = pgabc.create_loss(SquareLoss, parameters, self.ode, self.x0, self.t[0],
                                  self.t[1::], y, ['I', 'R'])
        sir_abc = pgabc.ABC(sir_obj, parameters)
        sir_abc.get_posterior_sample(N=100, tol=np.inf, G=10, q=0.5, M=50)
        sir_abc.continue_posterior_sample(N=100, tol=sir_abc.next_tol, G=10, q=0.5, M=50)
        med_est = np.median(sir_abc.res, axis=0)
        self.assertTrue(np.allclose(med_est, self.target, 1e-2, 1e-2))
        
        
    def test_SIR_abc_NormalLoss(self):
        y = self.solution[1::, 1:3]
        parameters = [pgabc.Parameter('beta', 'unif', 0, 3, logscale=False), 
                      pgabc.Parameter('gamma', 'unif', 0, 3, logscale=False)]
        sir_obj = pgabc.create_loss(NormalLoss, parameters, self.ode, self.x0, self.t[0],
                                  self.t[1::], y, ['I', 'R'])
        sir_abc = pgabc.ABC(sir_obj, parameters)
        sir_abc.get_posterior_sample(N=100, tol=np.inf, G=10, q=0.5)
        sir_abc.continue_posterior_sample(N=100, tol=sir_abc.next_tol, G=10, q=0.5)
        med_est = np.median(sir_abc.res, axis=0)
        self.assertTrue(np.allclose(med_est, self.target, 1e-2, 1e-2))
        
        
    def tearDown(self):
        self.ode = None
        self.x0 = None
        self.t = None
        self.solution = None
        self.target = None       
        
        
if __name__ == '__main__':
    main()
