from unittest import TestCase

import numpy
import scipy.integrate, scipy.optimize
import copy

from pygom import common_models, SquareLoss, NormalLoss, PoissonLoss
from pygom import Transition, TransitionType, OperateOdeModel

class TestModelEstimate(TestCase):

    def test_SIR_Estimate_SquareLoss(self):
        # define the model and parameters
        ode = common_models.SIR({'beta':0.5,'gamma':1.0/3.0})

        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = numpy.linspace(0, 150, 100)
        # Standard.  Find the solution.
        solution = scipy.integrate.odeint(ode.ode, x0, t)

        # y = copy.copy(solution[:,1:3])
        # initial value
        theta = [0.2, 0.2]

        # test out whether the single state function 'ok'
        objSIR = SquareLoss(theta, ode, x0, t[0], t[1::],
                            solution[1::,2], 'R')
        objSIR.cost()
        objSIR.gradient()
        objSIR.hessian()

        # now we go on the real shit
        objSIR = SquareLoss(theta, ode, x0, t[0], t[1::],
                            solution[1::,1:3], ['I','R'])

        # constraints
        EPSILON = numpy.sqrt(numpy.finfo(numpy.float).eps)

        boxBounds = [(EPSILON, 5), (EPSILON, 5)]

        resQP = scipy.optimize.minimize(fun=objSIR.cost,
                                        jac=objSIR.sensitivity,
                                        x0=theta,
                                        method='SLSQP',
                                        bounds=boxBounds)

        target = numpy.array([0.5, 1.0/3.0])
        if numpy.any(abs(resQP['x']-target) >= 1e-2):
            raise Exception("Failed!")

    def test_SIR_Estimate_SquareLoss_Adjoint(self):
        # define the model and parameters
        ode = common_models.SIR({'beta':0.5,'gamma':1.0/3.0})
        
        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = numpy.linspace(0, 150, 100)
        # Standard.  Find the solution.
        solution = scipy.integrate.odeint(ode.ode, x0, t)

        y = copy.copy(solution[:,1:3])
        # initial value
        theta = [0.2, 0.2]

        objSIR = SquareLoss(theta, ode, x0, t[0], t[1::], y[1::,:], ['I','R'])

        # constraints
        EPSILON = numpy.sqrt(numpy.finfo(numpy.float).eps)

        boxBounds = [(EPSILON, 5),(EPSILON, 5)]

        resQP = scipy.optimize.minimize(fun=objSIR.cost,
                                        jac=objSIR.adjoint,
                                        x0=theta,
                                        method='SLSQP',
                                        bounds=boxBounds)
        
        target = numpy.array([0.5, 1.0/3.0])
        if numpy.any(abs(resQP['x']-target) >= 1e-2):
            raise Exception("Failed!")

    def test_SIR_Estimate_NormalLoss(self):
        # define the model and parameters
        ode = common_models.SIR({'beta':0.5,'gamma':1.0/3.0})

        # the initial state, normalized to zero one
        x0 = [1, 1.27e-6, 0]
        # set the time sequence that we would like to observe
        t = numpy.linspace(0, 150, 100)
        # Standard.  Find the solution.
        solution = scipy.integrate.odeint(ode.ode, x0, t)

        y = copy.copy(solution[:,1:3])
        # initial value
        theta = [0.2, 0.2]

        objSIR = NormalLoss(theta, ode, x0, t[0], t[1::], y[1::,:], ['I','R'])

        # constraints
        EPSILON = numpy.sqrt(numpy.finfo(numpy.float).eps)
        
        boxBounds = [(EPSILON, 5), (EPSILON, 5)]

        resQP = scipy.optimize.minimize(fun=objSIR.cost,
                                        jac=objSIR.sensitivity,
                                        x0=theta,
                                        method='SLSQP',
                                        bounds=boxBounds)

        target = numpy.array([0.5, 1.0/3.0])
        if numpy.any(abs(resQP['x']-target) >= 1e-2):
            raise Exception("Failed!")

    def test_SIR_Estimate_PoissonLoss_1TargetState(self):
        # initial values
        N = 2362205.0
        x0 = [N, 3.0, 0.0]
        t = numpy.linspace(0, 150, 100).astype('float64')
        # params
        paramEval = [('beta', 0.5), ('gamma', 1.0/3.0),('N', N)]

        stateList = ['S','I','R']
        paramList = ['beta','gamma','N']
        transitionList = [
                          Transition(origState='S', destState='I',
                                     equation='beta * S * I/N',
                                     transitionType=TransitionType.T),
                          Transition(origState='I', destState='R',
                                     equation='gamma * I',
                                     transitionType=TransitionType.T)
                          ]
        # initialize the model
        ode = OperateOdeModel(stateList, paramList, transitionList=transitionList)
        ode = ode.setParameters(paramEval).setInitialValue(x0,t[0])

        # Standard.  Find the solution.
        solution = ode.integrate(t[1::])
        # initial value
        theta = [0.4,0.3]
        
        objSIR = PoissonLoss(theta, ode, x0, t[0], t[1::],
                             numpy.round(solution[1::,2]),
                             'R', targetParam=['beta','gamma'])

        # constraints
        EPSILON = numpy.sqrt(numpy.finfo(numpy.float).eps)
        
        boxBounds = [(EPSILON,2),(EPSILON,2)]

        res = scipy.optimize.minimize(fun=objSIR.cost,
                                      jac=objSIR.sensitivity,
                                      x0=theta,
                                      method='L-BFGS-B',
                                      bounds=boxBounds)
        
        target = numpy.array([0.5, 1.0/3.0])
        if numpy.any(abs(res['x']-target) >= 1e-2):
            raise Exception("Failed!")   

    def test_SIR_Estimate_PoissonLoss_2TargetState(self):
        # initial values
        N = 2362205.0
        x0 = [N, 3.0, 0.0]
        t = numpy.linspace(0, 150, 100).astype('float64')
        # params
        paramEval = [('beta', 0.5), ('gamma', 1.0/3.0),('N', N)]

        stateList = ['S','I','R']
        paramList = ['beta','gamma','N']
        transitionList = [
                          Transition(origState='S', destState='I',
                                     equation='beta * S * I/N',
                                     transitionType=TransitionType.T),
                          Transition(origState='I', destState='R',
                                     equation='gamma * I',
                                     transitionType=TransitionType.T)
                          ]
        # initialize the model
        ode = OperateOdeModel(stateList, paramList, transitionList=transitionList)
        ode = ode.setParameters(paramEval).setInitialValue(x0,t[0])

        # Standard.  Find the solution.
        solution = ode.integrate(t[1::])
        # initial value
        theta = [0.4,0.3]
        
        # note that we need to round the observations to integer for it to make sense
        objSIR = PoissonLoss(theta, ode, x0, t[0], t[1::],
                             numpy.round(solution[1::,1:3]),
                             ['I','R'], targetParam=['beta','gamma'])

        # constraints
        EPSILON = numpy.sqrt(numpy.finfo(numpy.float).eps)

        boxBounds = [(EPSILON, 2), (EPSILON, 2)]

        res = scipy.optimize.minimize(fun=objSIR.cost,
                                      jac=objSIR.sensitivity,
                                      x0=theta,
                                      method='L-BFGS-B',
                                      bounds=boxBounds)
        
        target = numpy.array([0.5, 1.0/3.0])
        if numpy.any(abs(res['x']-target) >= 1e-2):
            raise Exception("Failed!")   


    def test_FH_Obj(self):
        # initial values
        x0 = [-1.0, 1.0]
        t0 = 0
        # params
        paramEval = [('a', 0.2), ('b', 0.2),('c', 3.0)]

        ode = common_models.FitzHugh().setParameters(paramEval).setInitialValue(x0, t0)
        # the time points for our observations
        t = numpy.linspace(1, 20, 30).astype('float64')
        # Standard.  Find the solution which we will be used as "observations later"
        solution,output = ode.integrate(t, full_output=True)
        # initial guess
        theta = [0.5,0.5,0.5]

        #objFH = squareLoss(theta,ode,x0,t0,t,solution[1::,1],'R')
        objFH = SquareLoss(theta, ode, x0, t0, t, solution[1::,:], ['V','R'])
        
        g1 = objFH.adjoint(theta)
        #g2 = objFH.adjointInterpolate1(theta)
        #g3 = objFH.adjointInterpolate2(theta)
        g4 = objFH.sensitivity(theta)

        EPSILON = numpy.sqrt(numpy.finfo(numpy.float).eps)

        boxBounds = [
            (EPSILON, 5.0),
            (EPSILON, 5.0),
            (EPSILON, 5.0)
            ]

        res = scipy.optimize.minimize(fun=objFH.cost,
                                      jac=objFH.sensitivity,
                                      x0=theta,
                                      bounds=boxBounds,
                                      method='L-BFGS-B')

        res2 = scipy.optimize.minimize(fun=objFH.cost,
                                      jac=objFH.adjoint,
                                      x0=theta,
                                      bounds=boxBounds,
                                      method='L-BFGS-B')

        target = numpy.array([0.2, 0.2, 3.0])
        if numpy.any(abs(target-res['x']) >= 1e-2):
            raise Exception("Failed!")

        if numpy.any(abs(target-res2['x']) >= 1e-2):
            raise Exception("Failed!")


    def test_FH_IV(self):
        # initial values
        x0 = [-1.0, 1.0]
        t0 = 0
        # params
        paramEval = [('a', 0.2), ('b', 0.2),('c', 3.0)]

        ode = common_models.FitzHugh().setParameters(paramEval).setInitialValue(x0, t0)
        # the time points for our observations
        t = numpy.linspace(1, 20, 30).astype('float64')
        # Standard.  Find the solution.
        solution,output = ode.integrate(t, full_output=True)
        # initial guess
        theta = [0.5, 0.5, 0.5]

        objFH = SquareLoss(theta, ode, x0, t0, t, solution[1::,:], ['V','R'])
        
        EPSILON = numpy.sqrt(numpy.finfo(numpy.float).eps)

        boxBounds = [
            (EPSILON, 5.0),
            (EPSILON, 5.0),
            (EPSILON, 5.0),
            (None, None),
            (None, None)
            ]
        
        res = scipy.optimize.minimize(fun=objFH.costIV,
                                      jac=objFH.sensitivityIV,
                                      x0=theta + [-0.5,0.5],
                                      bounds=boxBounds,
                                      method='L-BFGS-B')
        
        target = numpy.array([0.2, 0.2, 3.0, -1.0, 1.0])
        if numpy.any(abs(target-res['x']) >= 1e-2):
            raise Exception("Failed!")
