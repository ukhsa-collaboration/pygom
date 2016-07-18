from unittest import TestCase

import numpy

from pygom import common_models, SquareLoss, NormalLoss

class TestLossTypes(TestCase):

    def test_FH_Square(self):
        # initial values
        x0 = [-1.0, 1.0]
        # params
        paramEval = [('a', 0.2), ('b', 0.2),('c', 3.0)]
        # the time points for our observations
        t = numpy.linspace(0, 20, 30).astype('float64')
        ode = common_models.FitzHugh().setParameters(paramEval).setInitialValue(x0, t[0])
        # Standard.  Find the solution which we will be used as "observations later"
        solution, output = ode.integrate(t[1::], full_output=True)
        # initial guess
        theta = [0.5, 0.5, 0.5]

        #objFH = squareLoss(theta,ode,x0,t0,t,solution[1::,1],'R')
        objFH = SquareLoss(theta, ode, x0, t[0], t[1::], solution[1::,:], ['V','R'])

        r = objFH.residual()

        # weight for each component
        w = [2.0, 3.0]

        s1 = 0
        for i in range(2): s1 += ((r[:,i]*w[i])**2).sum()

        objFH1 = SquareLoss(theta, ode, x0, t[0], t[1::], solution[1::,:],
                   ['V','R'], w)

        # now the weight is a vector
        w = numpy.random.rand(29, 2)
        objFH2 = SquareLoss(theta, ode, x0, t[0], t[1::], solution[1::,:],
                   ['V','R'], w)

        s2 = ((r * numpy.array(w))**2).sum()

        if abs(objFH1.cost() - s1) >= 1e-2:
            raise Exception("Failed!")
        
        if abs(objFH2.cost() - s2) >= 1e-2:
            raise Exception("Failed!")

    def test_FH_Normal(self):
        # initial values
        x0 = [-1.0, 1.0]
        t0 = 0
        # params
        paramEval = [('a', 0.2), ('b', 0.2),('c', 3.0)]

        ode = common_models.FitzHugh().setParameters(paramEval).setInitialValue(x0, t0)
        # the time points for our observations
        t = numpy.linspace(0, 20, 30).astype('float64')
        # Standard.  Find the solution which we will be used as "observations later"
        solution, output = ode.integrate(t[1::], full_output=True)
        # initial guess
        theta = [0.5, 0.5, 0.5]

        #objFH = squareLoss(theta,ode,x0,t0,t,solution[1::,1],'R')
        objFH = NormalLoss(theta, ode, x0, t[0], t[1::], solution[1::,:], ['V','R'])

        w = [2.0,3.0]
        objFH1 = NormalLoss(theta, ode, x0, t[0], t[1::], solution[1::,:],
                   ['V','R'], w)
       
        # now the weight is a vector
        w = numpy.random.rand(29, 2)
        objFH2 = NormalLoss(theta, ode, x0, t[0], t[1::], solution[1::,:],
                   ['V','R'], w)

        objFH.cost()
        objFH1.cost()
        objFH2.cost()

    def test_FH_Square_1State_Fail(self):
        totalFail = 0
        expectedFail = 4
        # initial values
        x0 = [-1.0, 1.0]
        # the time points for our observations
        t = numpy.linspace(0, 20, 30).astype('float64')
        # params
        paramEval = [('a', 0.2), ('b', 0.2),('c', 3.0)]

        ode = common_models.FitzHugh().setParameters(paramEval).setInitialValue(x0, t[0])

        # Standard.  Find the solution which we will be used as "observations later"
        solution, output = ode.integrate(t[1::], full_output=True)
        # initial guess
        theta = [0.5, 0.5, 0.5]

        wList = list()

        wList.append([-1.])
        wList.append([0])
        wList.append([2.0, 3.0])
        wList.append(numpy.random.rand(30))
        
        for w in wList:
            try:
                objFH = SquareLoss(theta, ode, x0, t[0], t[1::], solution[1::,:],
                                   'R', w)    
            except:
                totalFail += 1
            
        if totalFail != expectedFail:
            raise Exception("We passed some of the illegal input...")
        
    def test_FH_Square_2State_Fail(self):
        totalFail = 0
        expectedFail = 8
        # initial values
        x0 = [-1.0, 1.0]
        # the time points for our observations
        t = numpy.linspace(0, 20, 30).astype('float64')
        # params
        paramEval = [('a', 0.2), ('b', 0.2),('c', 3.0)]

        ode = common_models.FitzHugh().setParameters(paramEval).setInitialValue(x0, t[0])

        # Standard.  Find the solution which we will be used as "observations later"
        solution, output = ode.integrate(t[1::], full_output=True)
        # initial guess
        theta = [0.5, 0.5, 0.5]

        wList = list()

        wList.append([-2.0])
        wList.append([2.0, 3.0, 4.0])
        wList.append([0.0, 0.0])
        wList.append([1.0, -1.0])
        wList.append(numpy.random.rand(30))
        wList.append([numpy.random.rand(30), numpy.random.rand(31)])
        wList.append([numpy.random.rand(31), numpy.random.rand(31)])
        wList.append([numpy.random.rand(30), numpy.random.rand(30), numpy.random.rand(30)])

        for i, w in enumerate(wList):
            try:
                objFH = SquareLoss(theta, ode, x0, t[0], t[1::], solution[1::,:],
                                   ['V','R'], w)    
            except:
                print(i)
                totalFail += 1
            
        if totalFail != expectedFail:
            raise Exception("We passed some of the illegal input...")
