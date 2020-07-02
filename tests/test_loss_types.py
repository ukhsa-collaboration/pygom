from unittest import main, TestCase

import numpy as np


from pygom import Transition, TransitionType, SimulateOde, SquareLoss, NormalLoss, GammaLoss, PoissonLoss, NegBinomLoss
from pygom.utilR.distn import gamma_mu_shape, dnbinom
import copy

class Test_loss_classes(TestCase):

    def setUp(self):
        # initial values
        N = 1e6
        in_inf = 1
        self.init_state = [N - in_inf, in_inf, 0.0]
        # params
        self.param_eval = [('beta', 3.6), ('gamma', 0.2), ('N', N)]
        # the time points for our observations
        self.t = np.arange (0 , 51 , 0.25)
        states = ['S', 'I', 'R']
        params = ['beta', 'gamma', 'N']
        transitions = [Transition(origin='S', destination='I', equation='beta*S*I/N', 
                                  transition_type=TransitionType.T),
                       Transition(origin='I', destination='R', equation='gamma*I', 
                                  transition_type=TransitionType.T)]
        self.ode = SimulateOde(states, params, transition=transitions)
        self.ode.parameters = self.param_eval
        self.ode.initial_values = (self.init_state, self.t[0])

        # Standard.  Find the solution which we will be used as
        # "observations later"
        self.solution = self.ode.integrate(self.t[1:])
        # initial guess
        self.theta = [3.6, 0.2,N]
        self.yhat_ode=copy.deepcopy(self.ode)
        self.yhat_ode.parameters = [('beta', self.theta[0]), ('gamma', self.theta[1]), ('N', N)]
        self.yhat= self.yhat_ode.integrate(self.t[1:])
        
    def test_all_Loss_functions_produce_different_costs(self):
        Square_obj = SquareLoss(self.theta, self.ode, self.init_state, self.t[0],
                                self.t[1::], self.solution[1::,1:3], ['I', 'R'])
        Normal_obj = NormalLoss(self.theta, self.ode, self.init_state, self.t[0],
                                self.t[1::], self.solution[1::,1:3], ['I', 'R'])
        Gamma_obj = GammaLoss(self.theta, self.ode, self.init_state, self.t[0],
                                self.t[1::], self.solution[1::,1:3], ['I', 'R'])
        Poisson_obj = PoissonLoss(self.theta, self.ode, self.init_state, self.t[0],
                                self.t[1::], self.solution[1::,1:3], ['I', 'R'])
        NegBinom_obj = NegBinomLoss(self.theta, self.ode, self.init_state, self.t[0],
                                self.t[1::], self.solution[1::,1:3], ['I', 'R'])
        
        comparisons = [[Square_obj.cost(),Normal_obj.cost(),'SquareLoss compared to NormalLoss'],
                        [Square_obj.cost(),Gamma_obj.cost(),'SquareLoss compared to GammaLoss'],
                        [Square_obj.cost(),Poisson_obj.cost(),'SquareLoss compared to PossionLoss'],
                        [Square_obj.cost(),NegBinom_obj.cost(),'SquareLoss compared to NegBinomLoss'],
                        [Normal_obj.cost(),Gamma_obj.cost(),'NormalLoss compared to GammaLoss'],
                        [Normal_obj.cost(),Poisson_obj.cost(),'NormalLoss compared to PossionLoss'],
                        [Normal_obj.cost(),NegBinom_obj.cost(),'NormalLoss compared to NegBinomLoss'],
                        [Gamma_obj.cost(),Poisson_obj.cost(),'GammaLoss compared to PossionLoss'],
                        [Gamma_obj.cost(),NegBinom_obj.cost(),'GammaLoss compared to NegBinomLoss'],
                        [Poisson_obj.cost(),NegBinom_obj.cost(),'PoissonLoss compared to NegBinomLoss']
                        ]
        
        for comparison in comparisons:
            message = comparison[-1]
            with self.subTest(message):
                self.assertNotAlmostEqual(comparison[0],comparison[1],places=0)


    def test_Square_and_Normal_Loss_cost_scalar_weights_for_two_states(self):
        loss_functions = [SquareLoss, NormalLoss]
        y=self.solution[1::,1:3].flatten()
        yhat = self.yhat[1::,1:3].flatten()
        residual = y-yhat
        w = [2.0, 3.0]
        w_unraveled = np.array([w[0]*np.ones(self.solution[1::,1:3].shape[0]),
                                w[1]*np.ones(self.solution[1::,1:3].shape[0])]).transpose().flatten()
        residual =residual*w_unraveled        
        square_cost = (residual**2).sum()
        sigma= 1
        norm_logpdf_p1= -np.log(2)
        norm_logpdf_p2= np.log(2)/2
        norm_logpdf_p3= -np.log(np.pi)/2
        norm_logpdf_p4= np.log(1/sigma)
        norm_logpdf_p5_alt= -residual**2 / (2*sigma**2)
        norm_cost = (-(norm_logpdf_p1+norm_logpdf_p2+norm_logpdf_p3+
                        norm_logpdf_p4+norm_logpdf_p5_alt)).sum() 
        test_answers=[square_cost,norm_cost]
        
        for loss_function_index in range(len(loss_functions)):
            loss_function= loss_functions[loss_function_index]
            message = str(loss_function)
            with self.subTest(message):
                obj = loss_function(self.theta, self.ode, self.init_state, self.t[0],
                                    self.t[1::], self.solution[1::,1:3], ['I', 'R'], w)
                self.assertAlmostEqual(obj.cost(), test_answers[loss_function_index],places=1)

    def test_Square_and_Normal_Loss_cost_vector_weights_for_two_states(self):
        loss_functions = [SquareLoss, NormalLoss]
        y=self.solution[1::,1:3].flatten()
        yhat = self.yhat[1::,1:3].flatten()
        residual = y-yhat
        w = [2.0, 3.0]
        w = np.array([w[0]*np.ones(self.solution[1::,1:3].shape[0]),
                      w[1]*np.ones(self.solution[1::,1:3].shape[0])]).transpose()
        residual = residual*w.flatten()
        square_cost = (residual**2).sum()
        sigma= 1
        norm_logpdf_p1= -np.log(2)
        norm_logpdf_p2= np.log(2)/2
        norm_logpdf_p3= -np.log(np.pi)/2
        norm_logpdf_p4= np.log(1/sigma)
        norm_logpdf_p5_alt= -residual**2 / (2*sigma**2)
        norm_cost = (-(norm_logpdf_p1+norm_logpdf_p2+norm_logpdf_p3+
                        norm_logpdf_p4+norm_logpdf_p5_alt)).sum()
        
        test_answers=[square_cost,norm_cost]

        for loss_function_index in range(len(loss_functions)):
            loss_function= loss_functions[loss_function_index]
            message = str(loss_function)
            with self.subTest(message):
                obj = loss_function(self.theta, self.ode, self.init_state, self.t[0],
                                    self.t[1::], self.solution[1::,1:3], ['I', 'R'], w)
                self.assertAlmostEqual(obj.cost(), test_answers[loss_function_index],places=1)
                
       
    def test_All_Loss_functions_1State_weights_Failures_TypeErrors(self):
        loss_functions = [SquareLoss, NormalLoss, GammaLoss, PoissonLoss, NegBinomLoss]
        w_list = list()
        w_list.append('a')
        test_weight=np.ones(len(self.solution[1::,-1])).tolist()
        test_weight[-1]='b'
        w_list.append(test_weight)
        for loss_function in loss_functions:
            for w in w_list:
                message = str(loss_function)+' with weighting '+str(w)
                with self.subTest(message):
                    self.assertRaises(TypeError, loss_function, self.theta, self.ode,
                                      self.init_state, self.t[0], self.t[1::], self.solution[1::,-1],
                                      'R', w)
                
    def test_AllLoss_1State_weights_Failures_ValueErrors(self):
        loss_functions = [SquareLoss, NormalLoss, GammaLoss, PoissonLoss, NegBinomLoss]
        w_list = list()
        w_list.append(-1)
        w_list.append(0)
        test_weight=np.ones(len(self.solution[1::,-1]))
        w_list.append(-test_weight)
        test_weight[-1]=-1
        w_list.append(test_weight)
        w_list.append(np.zeros(len(self.solution[1::,-1])))
        for loss_function in loss_functions:
            for w in w_list:
                message = str(loss_function)+' with weighting '+str(w)
                with self.subTest(message):
                    self.assertRaises(ValueError, loss_function, self.theta, self.ode,
                                      self.init_state, self.t[0], self.t[1::], self.solution[1::,-1],
                                      'R', w)
        
    def test_AllLoss_1State_weights_Failures_ShapeErrors(self):
        loss_functions = [SquareLoss, NormalLoss, GammaLoss, PoissonLoss, NegBinomLoss]
        w_list = list()         
        w_list.append([2.0, 3.0])
        w_list.append(np.random.rand(self.solution[1::,-1].shape[0]+1).tolist())
        for loss_function in loss_functions:
            for w in w_list:
                message = str(loss_function)+' with weighting '+str(w)
                with self.subTest(message):
                    self.assertRaises(AssertionError, loss_function, self.theta, self.ode,
                                      self.init_state, self.t[0], self.t[1::], self.solution[1::,-1],
                                      'R', w)
    
    def test_AllLoss_2State_weights_Failures_TypeErrors(self):
        loss_functions = [SquareLoss, NormalLoss, GammaLoss, PoissonLoss, NegBinomLoss]
        w_list = list()
        w_list.append(['a','b'])
        test_weight=np.ones(self.solution[1::,1:3].shape).tolist()
        test_weight[-1][-1]='c'
        w_list.append(test_weight)
        for loss_function in loss_functions:
            for w in w_list:
                message = str(loss_function)+' with weighting '+str(w)
                with self.subTest(message):
                    self.assertRaises(TypeError, loss_function, self.theta, self.ode,
                                      self.init_state, self.t[0], self.t[1::], 
                                      self.solution[1::,1:3],['I', 'R'], w)
        
    def test_AllLoss_2State_weights_Failures_ValueErrors(self):
        loss_functions = [SquareLoss, NormalLoss, GammaLoss, PoissonLoss, NegBinomLoss]
        w_list = list()
        w_list.append(-1)
        w_list.append(0)
        test_weight=np.ones(self.solution[1::,1:3].shape)       
        w_list.append(-test_weight)
        test_weight[-1,-1]=-1
        w_list.append(test_weight)
        w_list.append(np.zeros(self.solution[1::,1:3].shape))
        w_list.append([1.0, -1.0])
        w_list.append([0, 0])
        for loss_function in loss_functions:
            for w in w_list:
                message = str(loss_function)+' with weighting '+str(w)
                with self.subTest(message):
                    self.assertRaises(ValueError, loss_function, self.theta, 
                                      self.ode,self.init_state, self.t[0], 
                                      self.t[1::], self.solution[1::,1:3],
                                      ['I', 'R'], w)
        
    def test_AllLoss_2State_weights_Failures_ShapeErrors(self):
        loss_functions = [SquareLoss, NormalLoss, GammaLoss, PoissonLoss, NegBinomLoss]
        w_list = list()        
        w_list.append([2.0, 3.0, 4.0])
        w_list.append(np.random.rand(self.solution[1::,1:3].shape[0]))
        w_list.append(np.random.rand(self.solution[1::,1:3].shape[0]+1,self.solution[1::,1:3].shape[1]).tolist())
        w_list.append(np.random.rand(self.solution[1::,1:3].shape[0],self.solution[1::,1:3].shape[1]+1).tolist())
        for loss_function in loss_functions:
            for w in w_list:
                message = str(loss_function)+' with weighting '+str(w)
                with self.subTest(message):
                    self.assertRaises(AssertionError, loss_function, self.theta,
                                      self.ode, self.init_state, self.t[0], 
                                      self.t[1::], self.solution[1::,1:3],
                                      ['I', 'R'], w)
    
    def test_Applicable_Loss_functions_cost_scalar_spread_params_for_two_states(self):
        loss_functions = [NormalLoss, GammaLoss, NegBinomLoss]
        y=self.solution[1::,1:3].flatten()
        yhat = self.yhat[1::,1:3].flatten()
        residual = y-yhat
        # spread parameter for each component
        spread_params = [0.5, 1.5]
        spread_params_unraveled = np.array([spread_params[0]*
                                            np.ones(self.solution[1::,1:3].shape[0]),
                                            spread_params[1]*
                                            np.ones(self.solution[1::,1:3].shape[0])]).transpose().flatten()
        norm_logpdf_p1= -np.log(2)
        norm_logpdf_p2= np.log(2)/2
        norm_logpdf_p3= -np.log(np.pi)/2
        norm_logpdf_p4= np.log(1/spread_params_unraveled)
        norm_logpdf_p5_alt= -residual**2 / (2*spread_params_unraveled**2)
        norm_cost = (-(norm_logpdf_p1+norm_logpdf_p2+norm_logpdf_p3+
                        norm_logpdf_p4+norm_logpdf_p5_alt)).sum()
        
        gamma_cost=(-gamma_mu_shape(x=y, mu=yhat,shape=spread_params_unraveled,log=True)).sum()
        NegBinom_cost=(-dnbinom(x=y, mu=yhat,size=spread_params_unraveled,log=True)).sum()
              
        test_answers=[norm_cost,gamma_cost,NegBinom_cost]
        
        for loss_function_index in range(len(loss_functions)):
            loss_function= loss_functions[loss_function_index]
            message = str(loss_function)
            with self.subTest(message):
                obj = loss_function(self.theta, self.ode, self.init_state, self.t[0],
                                    self.t[1::], self.solution[1::,1:3], ['I', 'R'], None,spread_params)
                self.assertAlmostEqual(obj.cost(), test_answers[loss_function_index],places=1)
    
    def test_Applicable_Loss_functions_cost_vector_spread_params_for_two_states(self):
        loss_functions = [NormalLoss, GammaLoss, NegBinomLoss]
        y=self.solution[1::,1:3].flatten()
        yhat = self.yhat[1::,1:3].flatten()
        residual = y-yhat
        # now the spread parameter is a vector
        spread_params = [0.5, 1.5]
        spread_params = np.array([spread_params[0]*
                                  np.ones(self.solution[1::,1:3].shape[0]),
                                  spread_params[1]*
                                  np.ones(self.solution[1::,1:3].shape[0])]).transpose()
        spread_params_unraveled=spread_params.flatten()
        norm_logpdf_p1= -np.log(2)
        norm_logpdf_p2= np.log(2)/2
        norm_logpdf_p3= -np.log(np.pi)/2
        norm_logpdf_p4= np.log(1/spread_params_unraveled)
        norm_logpdf_p5_alt= -residual**2 / (2*spread_params_unraveled**2)
        norm_cost = (-(norm_logpdf_p1+norm_logpdf_p2+norm_logpdf_p3+
                        norm_logpdf_p4+norm_logpdf_p5_alt)).sum()
        gamma_cost=(-gamma_mu_shape(x=y, mu=yhat,shape=spread_params_unraveled,log=True)).sum()
        NegBinom_cost=(-dnbinom(x=y, mu=yhat,size=spread_params_unraveled,log=True)).sum()
        
        test_answers=[norm_cost,gamma_cost,NegBinom_cost]
    
        for loss_function_index in range(len(loss_functions)):
            loss_function= loss_functions[loss_function_index]
            message = str(loss_function)
            with self.subTest(message):
                obj = loss_function(self.theta, self.ode, self.init_state, self.t[0],
                                    self.t[1::], self.solution[1::,1:3], ['I', 'R'], None,spread_params)
                self.assertAlmostEqual(obj.cost(), test_answers[loss_function_index],places=1)                           
       
    def test_Applicable_Loss_functions_1State_spread_param_Failures_TypeErrors(self):
        loss_functions = [NormalLoss, GammaLoss, NegBinomLoss]
        spread_param_list = list()
        spread_param_list.append('a')
        spread_param_list.append(True)
        test_spread_param=np.ones(len(self.solution[1::,-1])).tolist()
        test_spread_param[-1]='b'
        spread_param_list.append(test_spread_param)
        test_spread_param[-1]=False
        spread_param_list.append(test_spread_param)
        for loss_function in loss_functions:
            for spread_param in spread_param_list:
                message = str(loss_function)+' with spread params '+str(spread_param)
                with self.subTest(message):
                    self.assertRaises(TypeError, loss_function, self.theta, self.ode,
                                      self.init_state, self.t[0], self.t[1::], self.solution[1::,-1],
                                      'R', None,spread_param)
                
    def test_Applicable_Loss_functionss_1State_spread_param_Failures_ValueErrors(self):
        loss_functions = [NormalLoss, GammaLoss, NegBinomLoss]
        spread_param_list = list()
        spread_param_list.append(-1)
        test_spread_param=np.ones(len(self.solution[1::,-1]))
        spread_param_list.append(-test_spread_param)
        test_spread_param[-1]=-1
        spread_param_list.append(test_spread_param)
        for loss_function in loss_functions:
            for spread_param in spread_param_list:
                message = str(loss_function)+' with spread params '+str(spread_param)
                with self.subTest(message):
                    self.assertRaises(ValueError, loss_function, self.theta, self.ode,
                                      self.init_state, self.t[0], self.t[1::], self.solution[1::,-1],
                                      'R', None,spread_param)
        
    def test_Applicable_Loss_functions_1State_spread_param_Failures_ShapeErrors(self):
        loss_functions = [NormalLoss, GammaLoss, NegBinomLoss]
        spread_param_list = list()         
        spread_param_list.append([2.0, 3.0])
        spread_param_list.append(np.random.rand(self.solution[1::,-1].shape[0]+1).tolist())
        for loss_function in loss_functions:
            for spread_param in spread_param_list:
                message = str(loss_function)+' with spread params '+str(spread_param)
                with self.subTest(message):
                    self.assertRaises(AssertionError, loss_function, self.theta, self.ode,
                                      self.init_state, self.t[0], self.t[1::], self.solution[1::,-1],
                                      'R', None,spread_param)
    
    def test_Applicable_Loss_functions_2State_spread_param_Failures_TypeErrors(self):
        loss_functions = [NormalLoss, GammaLoss, NegBinomLoss]
        spread_param_list = list()
        spread_param_list.append('a')
        spread_param_list.append(True)
        spread_param_list.append(['a','b'])
        spread_param_list.append([True,False])
        test_spread_param=np.ones(self.solution[1::,1:3].shape).tolist()
        test_spread_param[-1][-1]='c'
        spread_param_list.append(test_spread_param)
        test_spread_param[-1][-1]=False
        spread_param_list.append(test_spread_param)
        for loss_function in loss_functions:
            for spread_param in spread_param_list:
                message = str(loss_function)+' with spread params '+str(spread_param)
                with self.subTest(message):
                    self.assertRaises(TypeError, loss_function, self.theta, self.ode,
                                      self.init_state, self.t[0], self.t[1::], 
                                      self.solution[1::,1:3],['I', 'R'], None,spread_param)
        
    def test_Applicable_Loss_functions_2State_spread_param_Failures_ValueErrors(self):
        loss_functions = [NormalLoss, GammaLoss, NegBinomLoss]
        spread_param_list = list()
        spread_param_list.append(-1)
        test_spread_param=np.ones(self.solution[1::,1:3].shape)       
        spread_param_list.append(-test_spread_param)
        test_spread_param[-1,-1]=-1
        spread_param_list.append(test_spread_param)
        spread_param_list.append([1.0, -1.0])
        spread_param_list.append([-1.0, -1.0])
        for loss_function in loss_functions:
            for spread_param in spread_param_list:
                message = str(loss_function)+' with spread params '+str(spread_param)
                with self.subTest(message):
                    self.assertRaises(ValueError, loss_function, self.theta, 
                                      self.ode,self.init_state, self.t[0], 
                                      self.t[1::], self.solution[1::,1:3],
                                      ['I', 'R'], None,spread_param)
        
    def test_Applicable_Loss_functions_2State_spread_param_Failures_ShapeErrors(self):
        loss_functions = [NormalLoss, GammaLoss, NegBinomLoss]
        spread_param_list = list()        
        spread_param_list.append([2.0, 3.0, 4.0])
        spread_param_list.append(np.random.rand(self.solution[1::,1:3].shape[0]))
        spread_param_list.append(np.random.rand(self.solution[1::,1:3].shape[0]+1,self.solution[1::,1:3].shape[1]).tolist())
        spread_param_list.append(np.random.rand(self.solution[1::,1:3].shape[0],self.solution[1::,1:3].shape[1]+1).tolist())
        for loss_function in loss_functions:
            for spread_param in spread_param_list:
                message = str(loss_function)+' with spread params '+str(spread_param)
                with self.subTest(message):
                    self.assertRaises(AssertionError, loss_function, self.theta,
                                      self.ode, self.init_state, self.t[0], 
                                      self.t[1::], self.solution[1::,1:3],
                                      ['I', 'R'], None,spread_param)

if __name__ == '__main__':
    main()
