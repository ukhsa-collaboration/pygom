from unittest import TestCase

import numpy as np
import scipy.stats

from pygom import SimulateOde, Transition, TransitionType
from pygom.utilR import rgamma
from pygom.model import common_models

class TestSIRStochasticModel(TestCase):

    def test_simulateParam1(self):
        '''
        Stochastic ode under the interpretation that the parameters follow
        some sort of distribution.  In this case, a scipy.distn object.
        '''
        t0 = 0
        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = np.linspace(0, 150, 100)
        # Standard.  Find the solution.
        ode = common_models.SIR()
        ode.parameters = [0.5, 1.0/3.0]
        ode.initial_values = (x0, t0)
        solutionReference = ode.integrate(t[1::], full_output=False)

        # now we need to define our ode explicitly
        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma']
        transition_list = [
                          Transition(origin='S', destination='I',
                                     equation='beta*S*I',
                                     transition_type=TransitionType.T),
                          Transition(origin='I', destination='R',
                                     equation='gamma*I',
                                     transition_type=TransitionType.T)
                          ]
        # our stochastic version
        odeS = SimulateOde(state_list, param_list,
                           transition=transition_list)

        # define our parameters in terms of two gamma distributions
        # where the expected values are the same as before [0.5,1.0/3.0]
        d = dict()
        d['beta'] = scipy.stats.gamma(100.0, 0.0, 1.0/200.0)
        d['gamma'] = scipy.stats.gamma(100.0, 0.0, 1.0/300.0)
        odeS.parameters = d
        odeS.initial_values = (x0, t0)

        # now we generate the solutions
        solutionDiff = odeS.simulate_param(t[1::], 1000) - solutionReference

        # test :)
        self.assertTrue(np.any(abs(solutionDiff) <= 0.2))

    def test_simulateParam2(self):
        '''
        Stochastic ode under the interpretation that the parameters follow
        some sort of distribution.  In this case, a function handle which
        has the same name as those found in R.
        '''
        t0 = 0
        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = np.linspace(0, 150, 100)
        # Standard.  Find the solution.
        ode = common_models.SIR()
        ode.parameters = [0.5, 1.0/3.0]
        ode.initial_values = (x0,t0)
        solutionReference = ode.integrate(t[1::], full_output=False)

        # now we need to define our ode explicitly
        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma']
        transition_list = [
                          Transition(origin='S', destination='I',
                                     equation='beta*S*I',
                                     transition_type=TransitionType.T),
                          Transition(origin='I', destination='R',
                                     equation='gamma*I',
                                     transition_type=TransitionType.T)
                          ]
        # our stochastic version
        odeS = SimulateOde(state_list, param_list,
                           transition=transition_list)

        # define our parameters in terms of two gamma distributions
        # where the expected values are the same as before [0.5,1.0/3.0]
        d = dict()
        d['beta'] = (rgamma,{'shape':100.0, 'rate':200.0})
        d['gamma'] = (rgamma,(100.0, 300.0))

        odeS.parameters = d
        odeS.initial_values = (x0, t0)

        # now we generate the solutions
        solutionDiff = odeS.simulate_param(t[1::], 1000) - solutionReference

        # test :)
        self.assertTrue(np.all(abs(solutionDiff) <= 0.2))

    def test_simulateParam_same_seed(self):
        '''
        Stochastic ode under the interpretation that the parameters follow
        some sort of distribution and simulating using the same seed
        should produce the same result.
        '''
        t0 = 0
        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = np.linspace(0, 150, 100)
        # Standard.  Find the solution.

        # now we need to define our ode explicitly
        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma']
        transition_list = [
                          Transition(origin='S', destination='I',
                                     equation='beta*S*I',
                                     transition_type=TransitionType.T),
                          Transition(origin='I', destination='R',
                                     equation='gamma*I',
                                     transition_type=TransitionType.T)
                          ]
        # our stochastic version
        odeS = SimulateOde(state_list, param_list,
                           transition=transition_list)

        # define our parameters in terms of two gamma distributions
        # where the expected values are the same as before [0.5,1.0/3.0]
        d = dict()
        d['beta'] = scipy.stats.gamma(100.0, 0.0, 1.0/200.0)
        d['gamma'] = scipy.stats.gamma(100.0, 0.0, 1.0/300.0)
        odeS.parameters = d
        odeS.initial_values = (x0, t0)

        # now we generate the solutions
        seed = np.random.randint(1000)
        np.random.seed(seed)
        solution1, Yall1 = odeS.simulate_param(t[1::], 1000, full_output=True)
        np.random.seed(seed)
        solution2, Yall2 = odeS.simulate_param(t[1::], 1000, full_output=True)

        self.assertTrue(np.allclose(solution1, solution2))

        for i, yi in enumerate(Yall1):
            self.assertTrue(np.allclose(Yall2[i], yi))


    def test_simulateParam_different_seed(self):
        '''
        Stochastic ode under the interpretation that the parameters follow
        some sort of distribution and simulating using different seeds
        should produce different results.
        '''
        t0 = 0
        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = np.linspace(0, 150, 100)
        # Standard.  Find the solution.

        # now we need to define our ode explicitly
        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma']
        transition_list = [
                          Transition(origin='S', destination='I',
                                     equation='beta*S*I',
                                     transition_type=TransitionType.T),
                          Transition(origin='I', destination='R',
                                     equation='gamma*I',
                                     transition_type=TransitionType.T)
                          ]
        # our stochastic version
        odeS = SimulateOde(state_list, param_list,
                           transition=transition_list)

        # define our parameters in terms of two gamma distributions
        # where the expected values are the same as before [0.5,1.0/3.0]
        d = dict()
        d['beta'] = scipy.stats.gamma(100.0, 0.0, 1.0/200.0)
        d['gamma'] = scipy.stats.gamma(100.0, 0.0, 1.0/300.0)
        odeS.parameters = d
        odeS.initial_values = (x0, t0)

        # now we generate the solutions
        np.random.seed(1)
        solution1, Yall1 = odeS.simulate_param(t[1::], 1000, full_output=True)
        np.random.seed(2)
        solution2, Yall2 = odeS.simulate_param(t[1::], 1000, full_output=True)

        self.assertFalse(np.allclose(solution1, solution2))

        for i, yi in enumerate(Yall1):
            self.assertFalse(np.allclose(Yall2, yi))


    def test_SimulateCTMC(self):
        '''
        Stochastic ode under the interpretation that we have a continuous
        time Markov chain as the underlying process
        '''
        #x0 = [1,1.27e-6,0] # original
        x0 = [2362206.0, 3.0, 0.0]
        t = np.linspace(0, 250, 50)
        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma', 'N']
        transition_list = [
                          Transition(origin='S', destination='I',
                                     equation='beta*S*I/N',
                                     transition_type=TransitionType.T),
                          Transition(origin='I', destination='R',
                                     equation='gamma*I',
                                     transition_type=TransitionType.T)
                          ]
        # initialize the model
        odeS = SimulateOde(state_list, param_list,
                           transition=transition_list)

        odeS.parameters = [0.5, 1.0/3.0, x0[0]]
        odeS.initial_values = (x0, t[0])
        solution = odeS.integrate(t[1::])
        odeS.transition_mean(x0, t[0])
        odeS.transition_var(x0, t[0])

        odeS.transition_mean(solution[10,:], t[10])
        odeS.transition_var(solution[10,:], t[10])

        _simX, _simT = odeS.simulate_jump(250, 3, full_output=True)

    def test_SimulateCTMC_same_seed(self):
        '''
        Testing that using the same seed produces the same simulation under
        a CTMC interpretation only under a serial simulation.  When simulting
        with a parallel backend, the result will be different as the seed
        does not propagate through.
        '''
        x0 = [2362206.0, 3.0, 0.0]
        t = np.linspace(0, 10, 10)
        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma', 'N']
        transition_list = [
                          Transition(origin='S', destination='I',
                                     equation='beta*S*I/N',
                                     transition_type=TransitionType.T),
                          Transition(origin='I', destination='R',
                                     equation='gamma*I',
                                     transition_type=TransitionType.T)
                          ]
        # initialize the model
        odeS = SimulateOde(state_list, param_list,
                           transition=transition_list)

        odeS.parameters = [0.5, 1.0/3.0, x0[0]]
        odeS.initial_values = (x0, t[0])
        solution = odeS.integrate(t[1::])

        seed = np.random.randint(1000)

        # ## First note that the default is a parallel simulation using
        # ## dask as the backend.  This does not use the seed
        # np.random.seed(seed)
        # simX1, simT1 = odeS.simulate_jump(t[1::], 10, full_output=True)
        # np.random.seed(seed)
        # simX2, simT2 = odeS.simulate_jump(t[1::], 10, full_output=True)
        
        # for i, xi in enumerate(simX1):
        #     self.assertFalse(np.allclose(simX2[i], xi))

        ## But if we run it in serial then the seed will be used
        ## and the output will be identical
        np.random.seed(seed)
        simX1, simT1 = odeS.simulate_jump(t[1::], 10,
                                          parallel=False, full_output=True)
        np.random.seed(seed)
        simX2, simT2 = odeS.simulate_jump(t[1::], 10,
                                          parallel=False, full_output=True)

        for i, xi in enumerate(simX1):
            self.assertTrue(np.allclose(simX2[i], xi))


    def test_SimulateCTMC_different_seed(self):
        '''
        Testing that using a different seed produces different simulations under
        a CTMC interpretation regardless of the backend.
        '''
        x0 = [2362206.0, 3.0, 0.0]
        t = np.linspace(0, 10, 10)
        state_list = ['S', 'I', 'R']
        param_list = ['beta', 'gamma', 'N']
        transition_list = [
                          Transition(origin='S', destination='I',
                                     equation='beta*S*I/N',
                                     transition_type=TransitionType.T),
                          Transition(origin='I', destination='R',
                                     equation='gamma*I',
                                     transition_type=TransitionType.T)
                          ]
        # initialize the model
        odeS = SimulateOde(state_list, param_list,
                           transition=transition_list)

        odeS.parameters = [0.5, 1.0/3.0, x0[0]]
        odeS.initial_values = (x0, t[0])
        solution = odeS.integrate(t[1::])

        np.random.seed(1)
        simX1, simT1 = odeS.simulate_jump(t[1::], 10, full_output=True)
        np.random.seed(2)
        simX2, simT2 = odeS.simulate_jump(t[1::], 10, full_output=True)

        for i, xi in enumerate(simX1):
            self.assertFalse(np.allclose(simX2[i], xi))

        np.random.seed(1)
        simX1, simT1 = odeS.simulate_jump(t[1::], 10,
                                          parallel=False, full_output=True)
        np.random.seed(2)
        simX2, simT2 = odeS.simulate_jump(t[1::], 10,
                                          parallel=False, full_output=True)

        for i, xi in enumerate(simX1):
            self.assertFalse(np.allclose(simX2[i], xi))

