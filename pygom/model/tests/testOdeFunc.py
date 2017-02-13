from unittest import TestCase

from pygom import common_models
import numpy
import scipy.integrate
import copy

class TestJacobians(TestCase):
    
    def test_odeJacobian(self):
        '''
        Analytic Jacobian for the ode against the forward
        differencing numeric Jacobian
        '''
        # initial time
        t0 = 0
        # the initial state, normalized to zero one
        x0 = [1,1.27e-6,0]
        # params
        paramEval = [('beta',0.5), ('gamma',1.0/3.0)]
        ode = common_models.SIR(paramEval).setInitialValue(x0,t0)
        
        d = ode.getNumState()
        p = ode.getNumParam()
        
        x0 = numpy.array(x0)
      
        t = numpy.linspace(0, 150, 100)
        # integrate without using the analytical Jacobian
        solution,output = scipy.integrate.odeint(ode.ode,
                                                 x0,t,
                                                 full_output=True)

        # the Jacobian of the ode itself
        h = numpy.sqrt(numpy.finfo(numpy.float).eps)
        index = 50
        # random.randomint(0,150)
        ff0 = solution[index,:]
        J0 = ode.ode(ff0,t[index])
        J = numpy.zeros((d,d))
        for i in range(0,d):
            for j in range(0,d):
                ffTemp = copy.deepcopy(ff0)
                ffTemp[j] += h
                J[i,j] = (ode.ode(ffTemp,t[index])[i] - J0[i]) / h
            
        JAnalytic = ode.Jacobian(ff0,t[index])
        if numpy.any(abs(J - JAnalytic)>=1e-4):
            raise Exception("Test Failed")

        
    def test_SensJacobian(self):
        '''
        Analytic Jacobian for the forward sensitivity equations against
        the forward differencing numeric Jacobian
        '''
        # initial time
        t0 = 0
        # the initial state, normalized to zero one
        x0 = [1,1.27e-6,0]
        # params
        paramEval = [('beta',0.5), ('gamma',1.0/3.0)]
        ode = common_models.SIR(paramEval).setInitialValue(x0,t0)
        
        d = ode.getNumState()
        p = ode.getNumParam()
        
        s0 = numpy.zeros(d*p)
        x0 = numpy.array(x0)
        ffParam = numpy.append(x0,s0)
        
        t = numpy.linspace(0, 150, 100)
        # integrate without using the analytical Jacobian
        solutionSens,outputSens = scipy.integrate.odeint(ode.odeAndSensitivity,
                                                       ffParam,t,
                                                       full_output=True)


        # the Jacobian of the ode itself
        h = numpy.sqrt(numpy.finfo(numpy.float).eps)
        index = 50
        # random.randomint(0,150)
        ff0 = solutionSens[index,:]
        J0 = ode.odeAndSensitivity(ff0,t[index])
        J = numpy.zeros((d*(p+1),d*(p+1)))
        for i in range(0,d*(p+1)):
            for j in range(0,d*(p+1)):
                ffTemp = copy.deepcopy(ff0)
                ffTemp[j] += h
                J[i,j] = (ode.odeAndSensitivity(ffTemp,t[index])[i] - J0[i]) / h
            
        JAnalytic = ode.odeAndSensitivityJacobian(ff0,t[index])
        if numpy.any(abs(J - JAnalytic)>=1e-4):
            raise Exception("Test Failed")
        
    def test_HessianJacobian(self):
        '''
        Analytic Jacobian for the forward foward sensitivity equations 
        i.e. the Hessian of the objective function against
        the forward differencing numeric Jacobian
        '''
        # initial time
        t0 = 0
        # the initial state, normalized to zero one
        x0 = [1,1.27e-6,0]
        # params
        paramEval = [('beta',0.5), ('gamma',1.0/3.0)]
        ode = common_models.SIR(paramEval).setInitialValue(x0,t0)
        d = ode.getNumState()
        p = ode.getNumParam()
        
        ff0 = numpy.zeros(d*p*p)
        s0 = numpy.zeros(d*p)
        x0 = numpy.array(x0)
        ffParam = numpy.append(numpy.append(x0,s0),ff0)
        # some small value
        h = numpy.sqrt(numpy.finfo(numpy.float).eps)
        # time frame
        t = numpy.linspace(0, 150, 100)
        # our integration
        solutionHessian,outputHessian = scipy.integrate.odeint(ode.odeAndForwardforward,
                                                       ffParam,t,full_output=True)
        
        numFF = len(ffParam)
        J = numpy.zeros((numFF,numFF))
        # define our target
        index = 50
        # random.randomint(0,150)
        # get the info
        ff0 = solutionHessian[index,:]
        # evaluate at target point
        J0 = ode.odeAndForwardforward(ff0,t[index])
        # the Analytical solution is 
        JAnalytic = ode.odeAndForwardforwardJacobian(ff0,t[index])
        # now we go and find the finite difference Jacobian
        for i in range(0,numFF):
            for j in range(0,numFF):
                ffTemp = copy.deepcopy(ff0)
                #ffTemp[i] += h
                ffTemp[j] += h
                J[i,j] = (ode.odeAndForwardforward(ffTemp,t[index])[i] - J0[i]) / h

        print(J - JAnalytic)
        # Note that the two Jacobian above are not equivalent.  Only block diagonal
        # is implemented in the analytic case
