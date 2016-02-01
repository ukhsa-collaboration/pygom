from unittest import TestCase

from pygom import OperateOdeModel, Transition, TransitionType
import numpy
import sympy
from collections import OrderedDict

## define parameters
paramEval = {'beta_00':0.0010107,'beta_01':0.0010107,'beta_10':0.0010107,'beta_11':0.0010107,
                 'd':0.02,'epsilon':45.6,'gamma':73.0,'N_0':10**6,'N_1':10**6,'p':0.01}

class TestModelCoupled(TestCase):
    
    def test_compareAll(self):
        '''
        Compare the solution of a coupled ode using three different
        ways of defining it
        '''
        
        ## naive version
        n = 2
        s = [str(i) for i in range(n)]

        beta = []
        lambdaStr = []
        lambdaName = []
        N,S,E,I,R = [],[],[],[],[]

        for i in s:
            N += ['N_'+i]
            S += ['S_'+i]
            E += ['E_'+i]
            I += ['I_'+i]
            R += ['R_'+i]
            lambdaTemp = '0 '
            for j in s: 
                beta += ['beta_'+i+j]
                if i==j:
                    lambdaTemp += '+ I_'+j+'* beta_'+i+j
                else:
                    lambdaTemp += '+ I_'+j+' * beta_'+i+j+ ' * p'
            lambdaStr += [lambdaTemp]
            lambdaName += ['lambda_'+i]

        paramList = beta + ['d','epsilon','gamma','p'] + N

        stateList = S+E+I+R

        transitionList = []
        bdList = []
        derivedParamList = []
        for i in range(n):
            derivedParamList += [(lambdaName[i],lambdaStr[i])]
            transitionList += [Transition(origState=S[i],destState=E[i],equation=lambdaName[i]+ '*' +S[i] ,transitionType=TransitionType.T)]
            transitionList += [Transition(origState=E[i],destState=I[i],equation=' epsilon * ' +E[i] ,transitionType=TransitionType.T)]
            transitionList += [Transition(origState=I[i],destState=R[i],equation=' gamma * ' +I[i] ,transitionType=TransitionType.T)]
            bdList += [Transition(origState=S[i], equation='d * '+S[i], transitionType=TransitionType.D)]
            bdList += [Transition(origState=E[i], equation='d * '+E[i], transitionType=TransitionType.D)]
            bdList += [Transition(origState=I[i], equation='d * '+I[i], transitionType=TransitionType.D)]
            bdList += [Transition(origState=R[i], equation='d * '+R[i], transitionType=TransitionType.D)]
            bdList += [Transition(origState=S[i], equation='d * '+N[i], transitionType=TransitionType.B)]
            
        ode = OperateOdeModel(stateList,
                              paramList,
                              derivedParamList=derivedParamList,
                              transitionList=transitionList,
                              birthDeathList=bdList)


        ## to find the stationary starting conditions
        for param in paramList:
            strAdd= param+' = sympy.symbols("' +param+ '")'
            exec(strAdd)

        N = sympy.symbols("N")

        R0 = (epsilon * N) / ( (d+epsilon) * (d+gamma) ) * (beta_00+beta_01)
        S = N / R0
        E = (d * N) / (d+epsilon) * (1-1/R0)
        I = (d*epsilon)/( (d+gamma)*(d+epsilon) ) * N * (1-1/R0)
        R = N - S - E - I

        paramEval1 = {'beta_00':0.0010107,'beta_01':0.0010107,'beta_10':0.0010107,'beta_11':0.0010107,
                      'd':0.02,'epsilon':45.6,'gamma':73.0,'N_0':10**6,'N_1':10**6,'N':10**6}

        x0 = [S.subs(paramEval1),E.subs(paramEval1),I.subs(paramEval1),R.subs(paramEval1)]

        t = numpy.linspace(0,40,100)
        x01 = []
        for s in x0:
            x01 += [s]
            x01 += [s]

        ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
        solution1 = ode.integrate(t[1::])


        ## shorter version
        n = 2
        s = [str(i) for i in range(n)]

        beta = []
        lambdaStr = []
        lambdaName = []

        stateName = ["S","E","I","R"]
        states = OrderedDict.fromkeys(stateName,[])
        N =  []

        for i in s:
            for v in states:
                states[v] = states[v]+[str(v)+"_"+i]
            N += ['N_'+i]
            lambdaTemp = '0'
            for j in s: 
                beta += ['beta_'+i+j]
                if i==j:
                    lambdaTemp += '+ I_'+j+'*beta_'+i+j
                else:
                    lambdaTemp += '+ I_'+j+'*beta_'+i+j+ ' * p'
            lambdaStr += [lambdaTemp]
            lambdaName += ['lambda_'+i]

        paramList = beta + ['d','epsilon','gamma','p'] + N

        stateList = []
        for v in states: stateList += states[v]

        transitionList = []
        bdList = []
        derivedParamList = []
        for i in range(n):
            derivedParamList += [(lambdaName[i],lambdaStr[i])]
            transitionList += [Transition(origState=states['S'][i],destState=states['E'][i],equation=lambdaName[i]+ '*' +states['S'][i] ,transitionType=TransitionType.T)]
            transitionList += [Transition(origState=states['E'][i],destState=states['I'][i],equation=' epsilon * ' +states['E'][i] ,transitionType=TransitionType.T)]
            transitionList += [Transition(origState=states['I'][i],destState=states['R'][i],equation=' gamma * ' +states['I'][i] ,transitionType=TransitionType.T)]
            bdList += [Transition(origState=states['S'][i], equation='d * '+states['S'][i], transitionType=TransitionType.D)]
            bdList += [Transition(origState=states['E'][i], equation='d * '+states['E'][i], transitionType=TransitionType.D)]
            bdList += [Transition(origState=states['I'][i], equation='d * '+states['I'][i], transitionType=TransitionType.D)]
            bdList += [Transition(origState=states['R'][i], equation='d * '+states['R'][i], transitionType=TransitionType.D)]
            bdList += [Transition(origState=states['S'][i], equation='d * '+N[i], transitionType=TransitionType.B)]

        ode = OperateOdeModel(stateList,
                              paramList,
                              derivedParamList=derivedParamList,
                              transitionList=transitionList,
                              birthDeathList=bdList)

        ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
        solution2 = ode.integrate(t[1::])

        ## even shorter version
        n = 2
        s = [str(i) for i in range(n)]
        
        beta = []
        lambdaStr = []
        lambdaName = []

        stateName = ["S","E","I","R"]
        states = OrderedDict.fromkeys(stateName,[])
        N =  []

        for i in s:
            for v in states:
                states[v] = states[v]+[str(v)+"_"+i]
            N += ['N_'+i]
            lambdaTemp = '0'
            for j in s: 
                beta += ['beta_'+i+j]
                if i==j:
                    lambdaTemp += '+ I_'+j+'*beta_'+i+j
                else:
                    lambdaTemp += '+ I_'+j+'*beta_'+i+j+ ' * p'
            lambdaStr += [lambdaTemp]
            lambdaName += ['lambda_'+i]

        paramList = beta + ['d','epsilon','gamma','p'] + N

        stateList = []
        for v in states: stateList += states[v]

        transitionList = []
        bdList = []
        derivedParamList = []
        for i in range(n):
            derivedParamList += [(lambdaName[i],lambdaStr[i])]
            transitionList += [Transition(origState=states['S'][i],destState=states['E'][i],equation=lambdaName[i]+ '*' +states['S'][i] ,transitionType=TransitionType.T)]
            transitionList += [Transition(origState=states['E'][i],destState=states['I'][i],equation=' epsilon * ' +states['E'][i] ,transitionType=TransitionType.T)]
            transitionList += [Transition(origState=states['I'][i],destState=states['R'][i],equation=' gamma * ' +states['I'][i] ,transitionType=TransitionType.T)]
            for v in states:
                bdList += [Transition(origState=states[v][i], equation='d * '+states[v][i], transitionType=TransitionType.D)]
            bdList += [Transition(origState=states['S'][i], equation='d * '+N[i], transitionType=TransitionType.B)]
            
        ode = OperateOdeModel(stateList,
                              paramList,
                              derivedParamList=derivedParamList,
                              transitionList=transitionList,
                              birthDeathList=bdList)

        ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
        solution3 = ode.integrate(t[1::])

        if numpy.any((solution1-solution2)>=0.1):
            raise Exception("Solution not match")
        else:
            print("happy")

        if numpy.any((solution3-solution2)>=0.1):
            raise Exception("Solution not match")
        else:
            print("happy")

