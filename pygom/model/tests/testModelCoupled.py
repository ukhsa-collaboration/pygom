from unittest import TestCase

from pygom import OperateOdeModel, Transition, TransitionType
import numpy
import sympy
from collections import OrderedDict
import six

## define parameters
paramEval = {'beta_00':0.0010107,'beta_01':0.0010107,'beta_10':0.0010107,'beta_11':0.0010107,
                 'd':0.02,'epsilon':45.6,'gamma':73.0,'N_0':10**6,'N_1':10**6,'p':0.01}

class TestModelCoupled(TestCase):
    
    def test_compareAll(self):
        '''
        Compare the solution of a coupled ode using different ways of defining it
        '''
        
        n = 2
        
        ###
        ### naive version
        ### 

#         n = 2
#         s = [str(i) for i in range(n)]
# 
#         beta = []
#         lambdaStr = []
#         lambdaName = []
#         N,S,E,I,R = [],[],[],[],[]
# 
#         for i in s:
#             N += ['N_'+i]
#             S += ['S_'+i]
#             E += ['E_'+i]
#             I += ['I_'+i]
#             R += ['R_'+i]
#             lambdaTemp = '0 '
#             for j in s: 
#                 beta += ['beta_'+i+j]
#                 if i==j:
#                     lambdaTemp += '+ I_'+j+'* beta_'+i+j
#                 else:
#                     lambdaTemp += '+ I_'+j+' * beta_'+i+j+ ' * p'
#             lambdaStr += [lambdaTemp]
#             lambdaName += ['lambda_'+i]
# 
#         paramList = beta + ['d','epsilon','gamma','p'] + N
# 
#         stateList = S+E+I+R
# 
#         transitionList = []
#         bdList = []
#         derivedParamList = []
#         for i in range(n):
#             derivedParamList += [(lambdaName[i],lambdaStr[i])]
#             transitionList += [Transition(origState=S[i],destState=E[i],equation=lambdaName[i]+ '*' +S[i] ,transitionType=TransitionType.T)]
#             transitionList += [Transition(origState=E[i],destState=I[i],equation=' epsilon * ' +E[i] ,transitionType=TransitionType.T)]
#             transitionList += [Transition(origState=I[i],destState=R[i],equation=' gamma * ' +I[i] ,transitionType=TransitionType.T)]
#             bdList += [Transition(origState=S[i], equation='d * '+S[i], transitionType=TransitionType.D)]
#             bdList += [Transition(origState=E[i], equation='d * '+E[i], transitionType=TransitionType.D)]
#             bdList += [Transition(origState=I[i], equation='d * '+I[i], transitionType=TransitionType.D)]
#             bdList += [Transition(origState=R[i], equation='d * '+R[i], transitionType=TransitionType.D)]
#             bdList += [Transition(origState=S[i], equation='d * '+N[i], transitionType=TransitionType.B)]
#             
#         ode = OperateOdeModel(stateList,
#                               paramList,
#                               derivedParamList=derivedParamList,
#                               transitionList=transitionList,
#                               birthDeathList=bdList)
# 
#         glb = globals()
#         ## to find the stationary starting conditions
#         for param in paramList:
#             # strAdd = param+' = sympy.symbols("' +param+ '")'
#             # six.exec_(strAdd)
#             glb[param] = sympy.symbols(param)
#             # print(strAdd)
#             
#             # exec(strAdd + " in glb, glb")
#             # glb[param] = exec(strAdd)
# 
#         # print(type(epsilon))
#         N = sympy.symbols("N")
# 
#         # exec('R0 = (epsilon*N)/( (d + epsilon)*(d + gamma) ) * (beta_00 + beta_01) in globals(), globals()')
# #         six.exec_("R0 = (epsilon*N)/( (d + epsilon)*(d + gamma) ) * (beta_00 + beta_01)")
# #         six.exec_("S = N/R0")
# #         six.exec_("E = (d*N) / (d + epsilon) * (1 - 1/R0)")
# #         six.exec_("I = (d*epsilon)/( (d + gamma)*(d + epsilon) )*N*(1 - 1/R0)")
# #         six.exec_("R = N - S - E - I")
#         R0 = (epsilon*N)/( (d + epsilon)*(d + gamma) ) * (beta_00 + beta_01)
#         S = N/R0
#         E = (d*N) / (d + epsilon) * (1 - 1/R0)
#         I = (d*epsilon)/( (d + gamma)*(d + epsilon) )*N*(1 - 1/R0)
#         R = N - S - E - I
# 
# 
#         paramEval1 = {'beta_00':0.0010107,'beta_01':0.0010107,'beta_10':0.0010107,'beta_11':0.0010107,
#                       'd':0.02,'epsilon':45.6,'gamma':73.0,'N_0':10**6,'N_1':10**6,'N':10**6}
# 
#         x0 = [S.subs(paramEval1),E.subs(paramEval1),I.subs(paramEval1),R.subs(paramEval1)]
# 
#         t = numpy.linspace(0,40,100)
#         x01 = []
#         for s in x0:
#             x01 += [s]
#             x01 += [s]
# 
#         ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
#         solution1 = ode.integrate(t[1::])

        solution1 = self.naive(n)
        # print(solution1)

        ###
        ### shorter version
        ### 
        
        solution2 = self.shorter(n)
        
#         if numpy.any((solution1 - solution2) >= 0.001):
#             raise Exception("Solution not match")
#         else:
#             print("pass")

#         n = 2
#         s = [str(i) for i in range(n)]
# 
#         beta = []
#         lambdaStr = []
#         lambdaName = []
# 
#         stateName = ["S","E","I","R"]
#         states = OrderedDict.fromkeys(stateName,[])
#         N =  []
# 
#         for i in s:
#             for v in states:
#                 states[v] = states[v]+[str(v)+"_"+i]
#             N += ['N_'+i]
#             lambdaTemp = '0'
#             for j in s: 
#                 beta += ['beta_'+i+j]
#                 if i==j:
#                     lambdaTemp += '+ I_'+j+'*beta_'+i+j
#                 else:
#                     lambdaTemp += '+ I_'+j+'*beta_'+i+j+ ' * p'
#             lambdaStr += [lambdaTemp]
#             lambdaName += ['lambda_'+i]
# 
#         paramList = beta + ['d','epsilon','gamma','p'] + N
# 
#         stateList = []
#         for v in states: stateList += states[v]
# 
#         transitionList = []
#         bdList = []
#         derivedParamList = []
#         for i in range(n):
#             derivedParamList += [(lambdaName[i],lambdaStr[i])]
#             transitionList += [Transition(origState=states['S'][i],destState=states['E'][i],equation=lambdaName[i]+ '*' +states['S'][i] ,transitionType=TransitionType.T)]
#             transitionList += [Transition(origState=states['E'][i],destState=states['I'][i],equation=' epsilon * ' +states['E'][i] ,transitionType=TransitionType.T)]
#             transitionList += [Transition(origState=states['I'][i],destState=states['R'][i],equation=' gamma * ' +states['I'][i] ,transitionType=TransitionType.T)]
#             bdList += [Transition(origState=states['S'][i], equation='d * '+states['S'][i], transitionType=TransitionType.D)]
#             bdList += [Transition(origState=states['E'][i], equation='d * '+states['E'][i], transitionType=TransitionType.D)]
#             bdList += [Transition(origState=states['I'][i], equation='d * '+states['I'][i], transitionType=TransitionType.D)]
#             bdList += [Transition(origState=states['R'][i], equation='d * '+states['R'][i], transitionType=TransitionType.D)]
#             bdList += [Transition(origState=states['S'][i], equation='d * '+N[i], transitionType=TransitionType.B)]
# 
#         ode = OperateOdeModel(stateList,
#                               paramList,
#                               derivedParamList=derivedParamList,
#                               transitionList=transitionList,
#                               birthDeathList=bdList)
# 
#         ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
#         solution2 = ode.integrate(t[1::])

        ###
        ### even shorter version
        ### 
        
        solution3 = self.even_shorter(n)
        
#         if numpy.any((solution3 - solution2) >= 0.001):
#             raise Exception("Solution not match")
#         else:
#             print("pass")
        
#         if numpy.any((solution3-solution2) >= 0.001):
#             raise Exception("Solution not match")
#         else:
#             print("pass")
# 
#         n = 2
#         s = [str(i) for i in range(n)]
#         
#         beta = []
#         lambdaStr = []
#         lambdaName = []
# 
#         stateName = ["S","E","I","R"]
#         states = OrderedDict.fromkeys(stateName,[])
#         N =  []
# 
#         for i in s:
#             for v in states:
#                 states[v] = states[v]+[str(v)+"_"+i]
#             N += ['N_'+i]
#             lambdaTemp = '0'
#             for j in s: 
#                 beta += ['beta_'+i+j]
#                 if i==j:
#                     lambdaTemp += '+ I_'+j+'*beta_'+i+j
#                 else:
#                     lambdaTemp += '+ I_'+j+'*beta_'+i+j+ ' * p'
#             lambdaStr += [lambdaTemp]
#             lambdaName += ['lambda_'+i]
# 
#         paramList = beta + ['d','epsilon','gamma','p'] + N
# 
#         stateList = []
#         for v in states: stateList += states[v]
# 
#         transitionList = []
#         bdList = []
#         derivedParamList = []
#         for i in range(n):
#             derivedParamList += [(lambdaName[i],lambdaStr[i])]
#             transitionList += [Transition(origState=states['S'][i],destState=states['E'][i],equation=lambdaName[i]+ '*' +states['S'][i] ,transitionType=TransitionType.T)]
#             transitionList += [Transition(origState=states['E'][i],destState=states['I'][i],equation=' epsilon * ' +states['E'][i] ,transitionType=TransitionType.T)]
#             transitionList += [Transition(origState=states['I'][i],destState=states['R'][i],equation=' gamma * ' +states['I'][i] ,transitionType=TransitionType.T)]
#             for v in states:
#                 bdList += [Transition(origState=states[v][i], equation='d * '+states[v][i], transitionType=TransitionType.D)]
#             bdList += [Transition(origState=states['S'][i], equation='d * '+N[i], transitionType=TransitionType.B)]
#             
#         ode = OperateOdeModel(stateList,
#                               paramList,
#                               derivedParamList=derivedParamList,
#                               transitionList=transitionList,
#                               birthDeathList=bdList)
# 
#         ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
#         solution3 = ode.integrate(t[1::])

        ###
        ### very short version
        ###
        solution4 = self.very_short(n)
        
#         if numpy.any((solution4 - solution3) >= 0.001):
#             raise Exception("Solution not match")
#         else:
#             print("pass")
            
#         n = 2
# 
#         beta = []
#         lambdaStr = []
#         lambdaName = []
# 
#         lcl = locals()
#         stateName = ['N','S','E','I','R']
#         for s in stateName:
#             # six.exec_('%s = %s' % (s, [s+'_'+str(i) for i in range(n)]))
#             glb[s] = [s+'_'+str(i) for i in range(n)]
#             lcl[s] = [s+'_'+str(i) for i in range(n)]
#         print(glb.keys())
#         print(lcl.keys())
#         
#         for i in range(n):
#             lambdaTemp = '0 '
#             for j in range(n):
#                 beta.append('beta_%s%s' % (i,j))
#                 lambdaTemp += '+ I_%s * beta_%s%s ' % (j, i, j)
#                 if i != j:
#                     lambdaTemp += ' * p'
#             lambdaStr += [lambdaTemp]
#             lambdaName += ['lambda_'+str(i)]
# 
#         paramList = beta + ['d','epsilon','gamma','p'] + N
# 
#         stateList = S+E+I+R
#         
#         transitionList = []
#         bdList = []
#         derivedParamList = []
#         for i in range(n):
#             derivedParamList += [(lambdaName[i],lambdaStr[i])]
#             transitionList += [Transition(origState=S[i],destState=E[i],equation=lambdaName[i]+ '*' +S[i] ,transitionType=TransitionType.T)]
#             transitionList += [Transition(origState=E[i],destState=I[i],equation=' epsilon * ' +E[i] ,transitionType=TransitionType.T)]
#             transitionList += [Transition(origState=I[i],destState=R[i],equation=' gamma * ' +I[i] ,transitionType=TransitionType.T)]
#     
#             bdList += [Transition(origState=S[i], equation='d * '+N[i], transitionType=TransitionType.B)]
#         for s in stateList:
#             bdList += [Transition(origState=s, equation='d * '+s, transitionType=TransitionType.D)]
# 
#         ode = OperateOdeModel(stateList,
#                               paramList,
#                               derivedParamList=derivedParamList,
#                               transitionList=transitionList,
#                               birthDeathList=bdList)
#         
#         ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
#         solution4 = ode.integrate(t[1::])
        
        ###
        ### confused version
        ### 
        solution5 = self.confused(n)
        
#         if numpy.any((solution5 - solution4) >= 0.001):
#             raise Exception("Solution not match")
#         else:
#             print("pass")

#         n = 2
#         stateName = ['N','S','E','I','R']
#         for s in stateName:
#             six.exec_('%s = %s' % (s, [s+'_'+str(i) for i in range(n)]))
# 
#         beta = []
#         bdList = list()
#         transitionList = list()
#         derivedParamList = list()
#         for i in range(n):
#             lambdaStr = '0 '
#             for j in range(n):
#                 beta.append('beta_%s%s' % (i,j))
#                 lambdaStr += '+ I_%s * beta_%s%s ' % (j, i, j)
#                 if i != j:
#                     lambdaStr += ' * p'
#             derivedParamList += [('lambda_'+str(i), lambdaStr)]
# 
#             transitionList += [Transition(origState=S[i],destState=E[i],equation='lambda_'+str(i)+ '*' +S[i] ,transitionType=TransitionType.T)]
#             transitionList += [Transition(origState=E[i],destState=I[i],equation=' epsilon * ' +E[i] ,transitionType=TransitionType.T)]
#             transitionList += [Transition(origState=I[i],destState=R[i],equation=' gamma * ' +I[i] ,transitionType=TransitionType.T)]
#             bdList += [Transition(origState=S[i], equation='d * '+N[i], transitionType=TransitionType.B)]
# 
#         stateList = S+E+I+R
#         for s in stateList:
#             bdList += [Transition(origState=s, equation='d * '+s, transitionType=TransitionType.D)]
# 
#         paramList = beta + ['d','epsilon','gamma','p'] + N
#             
#         ode = OperateOdeModel(stateList,
#                               paramList,
#                               derivedParamList=derivedParamList,
#                               transitionList=transitionList,
#                               birthDeathList=bdList)
#         
#         ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
#         solution5 = ode.integrate(t[1::])

        if numpy.any((solution1-solution2) >= 0.001):
            raise Exception("Solution not match")

        if numpy.any((solution3-solution2) >= 0.001):
            raise Exception("Solution not match")
            
        if numpy.any((solution4-solution3) >= 0.001):
            raise Exception("Solution not match")
        
        if numpy.any((solution5-solution4) >= 0.001):
            raise Exception("Solution not match")
        
    def naive(self, n):
        # n = 2
        s = [str(i) for i in range(n)]

        beta = []
        lambdaStr = []
        lambdaName = []
        N, S, E, I, R = [], [], [], [], []

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

        paramList = beta + ['d', 'epsilon', 'gamma', 'p'] + N

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

#         glb = globals()
#         ## to find the stationary starting conditions
#         for param in paramList:
#             # strAdd = param+' = sympy.symbols("' +param+ '")'
#             # six.exec_(strAdd)
#             glb[param] = sympy.symbols(param)
#             # print(strAdd)
#             
#             # exec(strAdd + " in glb, glb")
#             # glb[param] = exec(strAdd)
# 
#         # print(type(epsilon))
#         N = sympy.symbols("N")
# 
#         # exec('R0 = (epsilon*N)/( (d + epsilon)*(d + gamma) ) * (beta_00 + beta_01) in globals(), globals()')
# #         six.exec_("R0 = (epsilon*N)/( (d + epsilon)*(d + gamma) ) * (beta_00 + beta_01)")
# #         six.exec_("S = N/R0")
# #         six.exec_("E = (d*N) / (d + epsilon) * (1 - 1/R0)")
# #         six.exec_("I = (d*epsilon)/( (d + gamma)*(d + epsilon) )*N*(1 - 1/R0)")
# #         six.exec_("R = N - S - E - I")
#         R0 = (epsilon*N)/( (d + epsilon)*(d + gamma) ) * (beta_00 + beta_01)
#         S = N/R0
#         E = (d*N) / (d + epsilon) * (1 - 1/R0)
#         I = (d*epsilon)/( (d + gamma)*(d + epsilon) )*N*(1 - 1/R0)
#         R = N - S - E - I
# 
#         print(self.getInitialValue(paramList))
# 
#         paramEval1 = {'beta_00':0.0010107,'beta_01':0.0010107,'beta_10':0.0010107,'beta_11':0.0010107,
#                       'd':0.02,'epsilon':45.6,'gamma':73.0,'N_0':10**6,'N_1':10**6,'N':10**6}
# 
#         x0 = [S.subs(paramEval1),E.subs(paramEval1),I.subs(paramEval1),R.subs(paramEval1)]
# 
#         x01 = []
#         for s in x0:
#             x01 += [s]
#             x01 += [s]

        t = numpy.linspace(0,40,100)
        x01 = self.getInitialValue(paramList)
        
        ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
        solution1 = ode.integrate(t[1::])
        return(solution1)
    
    def shorter(self, n):
        # n = 2
        s = [str(i) for i in range(n)]

        beta = []
        lambdaStr = []
        lambdaName = []

        stateName = ["S", "E", "I", "R"]
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

        paramList = beta + ['d', 'epsilon', 'gamma', 'p'] + N

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
 
        t = numpy.linspace(0,40,100)
        x01 = self.getInitialValue(paramList)

        ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
        solution2 = ode.integrate(t[1::])

        return(solution2)
    
    def even_shorter(self, n):
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

        paramList = beta + ['d', 'epsilon', 'gamma', 'p'] + N

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

        t = numpy.linspace(0,40,100)
        x01 = self.getInitialValue(paramList)

        ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
        solution3 = ode.integrate(t[1::])
        
        return(solution3)

    def very_short(self, n):
        beta = []
        lambdaStr = []
        lambdaName = []

        var_dict = globals()
        stateName = ['N', 'S', 'E', 'I', 'R']
        for s in stateName:
            # six.exec_('%s = %s' % (s, [s+'_'+str(i) for i in range(n)]))
            # glb[s] = [s+'_'+str(i) for i in range(n)]
            var_dict[s] = [s+'_'+str(i) for i in range(n)]
        # print(glb.keys())
        # print(lcl.keys())
        
        for i in range(n):
            lambdaTemp = '0 '
            for j in range(n):
                beta.append('beta_%s%s' % (i,j))
                lambdaTemp += '+ I_%s * beta_%s%s ' % (j, i, j)
                if i != j:
                    lambdaTemp += ' * p'
            lambdaStr += [lambdaTemp]
            lambdaName += ['lambda_'+str(i)]

        paramList = beta + ['d', 'epsilon', 'gamma', 'p'] + N

        stateList = S + E + I + R
        
        transitionList = []
        bdList = []
        derivedParamList = []
        for i in range(n):
            derivedParamList += [(lambdaName[i],lambdaStr[i])]
            transitionList += [Transition(origState=S[i],destState=E[i],equation=lambdaName[i]+ '*' +S[i] ,transitionType=TransitionType.T)]
            transitionList += [Transition(origState=E[i],destState=I[i],equation=' epsilon * ' +E[i] ,transitionType=TransitionType.T)]
            transitionList += [Transition(origState=I[i],destState=R[i],equation=' gamma * ' +I[i] ,transitionType=TransitionType.T)]
    
            bdList += [Transition(origState=S[i], equation='d * '+N[i], transitionType=TransitionType.B)]
        for s in stateList:
            bdList += [Transition(origState=s, equation='d * '+s, transitionType=TransitionType.D)]

        ode = OperateOdeModel(stateList,
                              paramList,
                              derivedParamList=derivedParamList,
                              transitionList=transitionList,
                              birthDeathList=bdList)
        
        t = numpy.linspace(0,40,100)
        x01 = self.getInitialValue(paramList)

        ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
        solution4 = ode.integrate(t[1::])
        
        return(solution4)
    
    def confused(self, n):
        stateName = ['N', 'S', 'E', 'I', 'R']
#         for s in stateName:
#             six.exec_('%s = %s' % (s, [s+'_'+str(i) for i in range(n)]))
        var_dict = globals()
        stateName = ['N', 'S', 'E', 'I', 'R']
        for s in stateName:
            # six.exec_('%s = %s' % (s, [s+'_'+str(i) for i in range(n)]))
            # glb[s] = [s+'_'+str(i) for i in range(n)]
            var_dict[s] = [s+'_'+str(i) for i in range(n)]


        beta = []
        bdList = list()
        transitionList = list()
        derivedParamList = list()
        for i in range(n):
            lambdaStr = '0 '
            for j in range(n):
                beta.append('beta_%s%s' % (i,j))
                lambdaStr += '+ I_%s * beta_%s%s ' % (j, i, j)
                if i != j:
                    lambdaStr += ' * p'
            derivedParamList += [('lambda_'+str(i), lambdaStr)]

            transitionList += [Transition(origState=S[i],destState=E[i],equation='lambda_'+str(i)+ '*' +S[i] ,transitionType=TransitionType.T)]
            transitionList += [Transition(origState=E[i],destState=I[i],equation=' epsilon * ' +E[i] ,transitionType=TransitionType.T)]
            transitionList += [Transition(origState=I[i],destState=R[i],equation=' gamma * ' +I[i] ,transitionType=TransitionType.T)]
            bdList += [Transition(origState=S[i], equation='d * '+N[i], transitionType=TransitionType.B)]

        stateList = S + E + I + R
        for s in stateList:
            bdList += [Transition(origState=s, equation='d * '+s, transitionType=TransitionType.D)]

        paramList = beta + ['d', 'epsilon', 'gamma', 'p'] + N
            
        ode = OperateOdeModel(stateList,
                              paramList,
                              derivedParamList=derivedParamList,
                              transitionList=transitionList,
                              birthDeathList=bdList)

        t = numpy.linspace(0,40,100)
        x01 = self.getInitialValue(paramList)

        ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
        solution5 = ode.integrate(t[1::])
        
        return(solution5)
        
    def getInitialValue(self, paramList):
        var_dict = globals()
        ## to find the stationary starting conditions
        for param in paramList:
            var_dict[param] = sympy.symbols(param)

        N = sympy.symbols("N")

        R0 = (epsilon*N)/( (d + epsilon)*(d + gamma) ) * (beta_00 + beta_01)
        S = N/R0
        E = (d*N) / (d + epsilon) * (1 - 1/R0)
        I = (d*epsilon)/( (d + gamma)*(d + epsilon) )*N*(1 - 1/R0)
        R = N - S - E - I

        paramEval1 = {'beta_00':0.0010107,'beta_01':0.0010107,'beta_10':0.0010107,'beta_11':0.0010107,
                      'd':0.02,'epsilon':45.6,'gamma':73.0,'N_0':10**6,'N_1':10**6,'N':10**6}

        x0 = [S.subs(paramEval1),E.subs(paramEval1),I.subs(paramEval1),R.subs(paramEval1)]
        x01 = []
        for s in x0:
            x01 += [s]
            x01 += [s]

        return(x01)