from unittest import main, TestCase

import sympy



from pygom import Transition, ODEVariable, DeterministicOde
import matplotlib.pyplot as plt

import numpy 

class TestEpiAnalysis(TestCase):

   def test_kalman_filter_result(self):
        """
        This test is based on the Demo Notebook.It tests if the KF can identify
        the beta value which was used to create a set of observations
        
        """
        
        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma','N']
        transition_list = [
                          Transition(origin='S', destination='I', equation='beta*S*I/N', transition_type="T"),
                          Transition(origin='I', destination='R', equation='gamma*I', transition_type="T")
                         ]
       # ode_stochastic = SimulateOde(state_list, param_list, transition=transition_list)
        beta_start =1.8
        d = dict()
        d['beta']  = beta_start
        d['gamma'] = 0.2
        d['N']     = 1000
        
        t1 = 100
        N = 1000
        
        x0=  [ 900.0  , 100.0  ,  0.0]
        dt=0.25
        x=[x0]
        t0 = 0
     
        states = ['S','I','R']
        extended_ode_deterministic= DeterministicOde(states, d, transition=transition_list)
        extended_ode_deterministic.parameters =d
        t=t0
        
        #try a test of the new KalmanIntegrate method
        t = numpy.linspace(t0, 100, 1000)
        extended_ode_deterministic.initial_values = (x0, t[0])
        # create a lsit of observations
        Outputs = extended_ode_deterministic.integrate(t)
        
        
        observations = numpy.random.uniform(low=0.0, high=10.0, size=len(t))
        observations= [Output[1] for Output in Outputs]
        observations =observations[2:]
        t =t[:len(observations)]
        
        #observations = observations.tolist()
        # set everything up again ,but this time with beta as a variable
        state_list = ['S', 'I', 'R','beta']
        param_list = [ 'gamma','N']
        transition_list = [
                          Transition(origin='S', destination='I', equation='beta*S*I/N', transition_type="T"),
                          Transition(origin='I', destination='R', equation='gamma*I', transition_type="T")
                         ]
        
        d = dict()
        
        d['gamma'] = 0.2
        d['N']     = 1000
        
        t1 = 100
        N = 1000
        
        x0=  [ 900.0  , 100.0  ,  0.0,10]
        dt=0.25
        x=[x0]
        t0 = 0
        
       
        states = ['S','I','R','beta']
        extended_ode_deterministic_beta= DeterministicOde(states, d, transition=transition_list)
        extended_ode_deterministic_beta.parameters =d
        #indices of states whcih are being observed
        observed_states =[1]
        sigmap2 =10.0
        sigmaw2 =1.0
        sigmar2 =1.0

        Outputs_beta =extended_ode_deterministic_beta.IntegrateWithKalmanFilter(extended_ode_deterministic_beta,x0,t, observations, observed_states, sigmap2,sigmaw2,sigmar2,beta_state=3)
        beta_values=[]
        for i in range(len(Outputs_beta)):
           beta_values.append(Outputs_beta[i][-1])
        beta_end = beta_values[-1]
       
        self.assertTrue(abs (beta_end-beta_start) < 0.05*beta_start)
        
  
        
   def test_Noisy_KF(self):     
    #  this ts make observation noise very large -does KF just return 
        """
        This test is based on the Demo Notebook.It tests if the KF can identify
        the beta value which was used to create a set of observations
      
        """
      
     
        transition_list = [
                          Transition(origin='S', destination='I', equation='beta*S*I/N', transition_type="T"),
                          Transition(origin='I', destination='R', equation='gamma*I', transition_type="T")
                         ]
       # ode_stochastic = SimulateOde(state_list, param_list, transition=transition_list)
        beta_start =1.8
        d = dict()
        d['beta']  = beta_start
        d['gamma'] = 0.2
        d['N']     = 1000
      
      
      
        x0=  [ 900.0  , 100.0  ,  0.0]
      
        t0 = 0
   
        states = ['S','I','R']
        extended_ode_deterministic= DeterministicOde(states, d, transition=transition_list)
        extended_ode_deterministic.parameters =d
        t=t0
      
        #try a test of the new KalmanIntegrate method
        t = numpy.linspace(t0, 100, 1000)
        extended_ode_deterministic.initial_values = (x0, t[0])
        # create a lsit of observations
        Outputs = extended_ode_deterministic.integrate(t)
      
      
        observations = numpy.random.uniform(low=0.0, high=10.0, size=len(t))
        observations= [Output[1] for Output in Outputs]
        observations =observations[2:]
        t =t[:len(observations)]
      
        #observations = observations.tolist()
        # set everything up again ,but this time with beta as a variable
      
        transition_list = [
                          Transition(origin='S', destination='I', equation='beta*S*I/N', transition_type="T"),
                          Transition(origin='I', destination='R', equation='gamma*I', transition_type="T")
                         ]
      
        d = dict()
      
        d['gamma'] = 0.2
        d['N']     = 1000
      
      
     
        x0=  [ 900.0  , 100.0  ,  0.0,10]
     
        t0 = 0
      
        #in this case we expect the Beta value to be unaltered because the data is very Noisy
        beta_start =10.0
        states = ['S','I','R','beta']
        extended_ode_deterministic_beta= DeterministicOde(states, d, transition=transition_list)
        extended_ode_deterministic_beta.parameters =d
        #indices of states whcih are being observed
        observed_states =[1]
        sigmap2 =10.0
        sigmaw2 =1.0
        # make observation noise very noisy
        sigmar2 =1.0e12
        
        extended_ode_deterministic_beta.initial_values = (x0, t[0])
      
        Outputs_Beta =extended_ode_deterministic_beta.IntegrateWithKalmanFilter(extended_ode_deterministic_beta,x0,t, observations, observed_states, sigmap2,sigmaw2,sigmar2,beta_state=3)
      
       
       
       
        beta_values=[]
        for i in range(len(Outputs_Beta)):
           beta_values.append(Outputs_Beta[i][-1])
        beta_end = beta_values[-1]
       
        self.assertTrue(abs (beta_end-beta_start) < 0.05*beta_start) 
        
   def test_Bounded_KF(self):     
         #  this ts make observation noise very large -does KF just return 
             """
             This test is based on the Demo Notebook.It tests if the KF can identify
             the beta value which was used to create a set of observations
           
             """
           
          
             transition_list = [
                               Transition(origin='S', destination='I', equation='beta*S*I/N', transition_type="T"),
                               Transition(origin='I', destination='R', equation='gamma*I', transition_type="T")
                              ]
            # ode_stochastic = SimulateOde(state_list, param_list, transition=transition_list)
             beta_start =1.8
             d = dict()
             d['beta']  = beta_start
             d['gamma'] = 0.2
             d['N']     = 1000
           
           
           
             x0=  [ 900.0  , 100.0  ,  0.0]
           
             t0 = 0
        
             states = ['S','I','R']
             extended_ode_deterministic= DeterministicOde(states, d, transition=transition_list)
             extended_ode_deterministic.parameters =d
             t=t0
           
             #try a test of the new KalmanIntegrate method
             t = numpy.linspace(t0, 100, 1000)
             extended_ode_deterministic.initial_values = (x0, t[0])
             # create a lsit of observations
             Outputs = extended_ode_deterministic.integrate(t)
           
           
             observations = numpy.random.uniform(low=0.0, high=10.0, size=len(t))
             observations= [Output[1] for Output in Outputs]
             observations =observations[2:]
             t =t[:len(observations)]
           
             #observations = observations.tolist()
             # set everything up again ,but this time with beta as a variable
           
             transition_list = [
                               Transition(origin='S', destination='I', equation='beta*S*I/N', transition_type="T"),
                               Transition(origin='I', destination='R', equation='gamma*I', transition_type="T")
                              ]
           
             d = dict()
           
             d['gamma'] = 0.2
             d['N']     = 1000
           
           
           
           
             x0=  [ 900.0  , 100.0  ,  0.0,10]
          
             t0 = 0
           
             #in this case we expect the Beta value to be unaltered because the data is very Noisy
             beta_start =10.0
             states = ['S','I','R','beta']
             extended_ode_deterministic_beta= DeterministicOde(states, d, transition=transition_list)
             extended_ode_deterministic_beta.parameters =d
             #indices of states whcih are being observed
             observed_states =[1]
             sigmap2 =10.0
             sigmaw2 =1.0
           
             sigmar2 =1.0
             # Becasue beta is bounded to 5.0 we should bec checking for 5.0
             beta_start=5.0
             extended_ode_deterministic_beta.initial_values = (x0, t[0])
             Bounds_List=[[None,None],[None,None],[None,None],[5.0,5.05]]
             Outputs_Beta =extended_ode_deterministic_beta.IntegrateWithKalmanFilter(extended_ode_deterministic_beta,x0,t, observations, observed_states, sigmap2,sigmaw2,sigmar2,Bounds= Bounds_List)
           
            
            
            
             beta_values=[]
             for i in range(len(Outputs_Beta)):
                beta_values.append(Outputs_Beta[i][-1])
             beta_end = beta_values[-1]
            
             self.assertTrue(abs (beta_end-beta_start) < 0.05*beta_start) 
             
  
                       
             
if __name__ == '__main__':
    main()
