from unittest import TestCase

from pygom import common_models, SimulateOdeModel, Transition, TransitionType
from pygom.utilR import rgamma
import scipy.stats
import numpy

class TestSIRStochasticModel(TestCase):

    def test_simulateParam1(self):
        '''
        Stochastic ode under the interpretation that the parameters follow
        some sort of distribution.  In this case, a scipy.distn object
        '''
        t0 = 0
        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = numpy.linspace(0, 150, 100)
        # Standard.  Find the solution.
        ode = common_models.SIR()
        ode = ode.setParameters([0.5, 1.0/3.0]).setInitialValue(x0, t0)
        solutionReference = ode.integrate(t[1::], full_output=False)
        
        # now we need to define our ode explicitly
        stateList = ['S', 'I', 'R']
        paramList = ['beta', 'gamma']
        transitionList = [
                          Transition(origState='S', destState='I',
                                     equation='beta*S*I',
                                     transitionType=TransitionType.T),
                          Transition(origState='I', destState='R',
                                     equation='gamma*I',
                                     transitionType=TransitionType.T)
                          ]
        # our stochastic version
        odeS = SimulateOdeModel(stateList,
                                paramList,
                                transitionList=transitionList)
        
        # define our parameters in terms of two gamma distributions
        # where the expected values are the same as before [0.5,1.0/3.0]
        d = dict()
        d['beta'] = scipy.stats.gamma(100.0, 0.0, 1.0/200.0)
        d['gamma'] = scipy.stats.gamma(100.0, 0.0, 1.0/300.0)
        odeS.setParameters(d).setInitialValue(x0, t0)    
        
        # now we generate the solutions
        solutionDiff = odeS.simulateParam(t[1::], 1000) - solutionReference
        
        # test :)
        if numpy.any(abs(solutionDiff) >= 0.2):
            raise Exception("Possible problem with simulating the parameters")
        
    def test_simulateParam2(self):
        '''
        Stochastic ode under the interpretation that the parameters follow
        some sort of distribution.  In this case, a function handle which
        has the same name as R
        '''
        t0 = 0
        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = numpy.linspace(0, 150, 100)
        # Standard.  Find the solution.
        ode = common_models.SIR()
        ode.setParameters([0.5, 1.0/3.0])
        ode.setInitialValue(x0,t0)
        solutionReference = ode.integrate(t[1::],full_output=False)
        
        # now we need to define our ode explicitly
        stateList = ['S', 'I', 'R']
        paramList = ['beta', 'gamma']
        transitionList = [
                          Transition(origState='S', destState='I',
                                     equation='beta*S*I',
                                     transitionType=TransitionType.T),
                          Transition(origState='I', destState='R',
                                     equation='gamma*I',
                                     transitionType=TransitionType.T)
                          ]
        # our stochastic version
        odeS = SimulateOdeModel(stateList,
                                paramList,
                                transitionList=transitionList)
        
        # define our parameters in terms of two gamma distributions
        # where the expected values are the same as before [0.5,1.0/3.0]
        d = dict()
        d['beta'] = (rgamma,{'shape':100.0, 'rate':200.0})
        d['gamma'] = (rgamma,(100.0, 300.0))
        
        odeS.setParameters(d).setInitialValue(x0, t0)
        
        # now we generate the solutions
        solutionDiff = odeS.simulateParam(t[1::], 1000) - solutionReference
        
        # test :)
        if numpy.any(abs(solutionDiff) >= 0.2):
            raise Exception("Possible problem with simulating the parameters")
        
    def test_SimulateCTMC(self):
        '''
        Stochastic ode under the interpretation that we have a continuous
        time Markov chain as the underlying process 
        '''
        #x0 = [1,1.27e-6,0] # original
        x0 = [2362206.0, 3.0, 0.0]
        t = numpy.linspace(0, 250, 50)
        stateList = ['S', 'I', 'R']
        paramList = ['beta', 'gamma', 'N']
        transitionList = [
                          Transition(origState='S', destState='I',
                                     equation='beta*S*I/N',
                                     transitionType=TransitionType.T),
                          Transition(origState='I', destState='R',
                                     equation='gamma*I',
                                     transitionType=TransitionType.T)
                          ]
        # initialize the model    
        odeS = SimulateOdeModel(stateList,
                                paramList,
                                transitionList=transitionList)

        odeS.setParameters([0.5, 1.0/3.0, x0[0]]).setInitialValue(x0, t[0])
        solution = odeS.integrate(t[1::])
        odeS.transitionMean(x0, t[0])
        odeS.transitionVar(x0, t[0])

        odeS.transitionMean(solution[10,:], t[10])
        odeS.transitionVar(solution[10,:], t[10])

        simX, simT = odeS.simulateJump(250, 3, full_output=True)
        
