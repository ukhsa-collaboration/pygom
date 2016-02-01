"""

    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    This module contains the classes required to translate inputs in string
    into an algebraic machine using sympy

"""
# Although reimporting * is not generally recommended
# we have to do it here so that it has all the mathematical
# functions ready to use when evaluating the equations.
# An alternative is to check for all the maths functions, such
# as exp, log, trigonometric etc.. and convert them to sympy
#from sympy.functions import (tan, cos, sin,asin, acos, atan, atan2, acot, cot, sec, csc)
from sympy.functions.elementary.trigonometric import (tan, cos, sin,
    asin, acos, atan, atan2, acot, cot, sec, csc)
from sympy.functions.elementary.exponential import (exp_polar, exp, log,
    LambertW)
from sympy.functions.combinatorial.factorials import (factorial, factorial2,
    rf, ff, binomial, RisingFactorial, FallingFactorial, subfactorial)
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.miscellaneous import (sqrt, root, Min, Max,
    Id, real_root, cbrt)
from sympy.core.numbers import pi

import sympy
from sympy import symbols
from sympy.simplify.simplify import simplify
from sympy.matrices import zeros
# string evaluation
import re
# numerical computation
import numpy
import scipy.stats

from .transition import Transition, TransitionType
from ._modelErrors import InputError, OutputError
import ode_utils

class BaseOdeModel(object):
    '''
    This contains the base that stores all the information of an ode

    Parameters
    ----------
    stateList: list
        A list of states (string)
    paramList: list
        A list of the parameters (string)
    derivedParamList: list
        A list of the derived parameters (tuple of (string,string))
    transitionList: list
        A list of transition (:class:`.Transition`)
    birthDeathList: list
        A list of birth or death process (:class:`.Transition`)
    odeList: list
        A list of ode (:class:`.Transition`)

    '''

    def __init__(self,
                 stateList=None,
                 paramList=None,
                 derivedParamList=None,
                 transitionList=None,
                 birthDeathList=None,
                 odeList=None):
        '''
        Constructor
        '''
        # the 3 required input when doing evaluation
        self._state = None
        self._param = None
        self._time = None
        # we always need time to be a symbol... and it should be denoted as t
        self._t = symbols('t')

        #
        self._isDifficult = False
        # allows the system to be defined directly
        self._odeList = list()
        self._explicitOde = False

        # book keeping parameters/states and etc
        self._paramList = list()
        # holder for the values of the parameters
        self._paramValue = None

        self._stateList = list()
        self._derivedParamList = list()

        # this three is not actually that useful
        # but lets leave it here for now
        self._numState = 0
        self._numParam = 0
        self._numDerivedParam = 0
        self._parameters = None
        self._stochasticParam = None

        self._hasNewTransition = False

        # dictionary for mapping
        self._paramDict = dict()
        # although time is not defined as a parameter, we want to keep
        # record of it existence
        self._paramDict['t'] = self._t
        self._stateDict = dict()
        self._derivedParamDict = dict()

        # dictionary to store vector symbols
        self._vectorStateDict = dict()

        #  holders for the actual equations
        self._transitionList = list()
        self._transitionMatrix = None
        self._birthDeathList = list()
        self._birthDeathVector = list()

        if stateList is not None:
            self.setStateList(stateList)

        if paramList is not None:
            self.setParamList(paramList)

        # this has to go after adding the parameters
        # because it is suppose to be based on the current
        # base parameters.
        # Making the distinction here because it makes a
        # difference when inferring the parameters of the variables
        if derivedParamList is not None:
            self.setDerivedParamList(derivedParamList)

        if transitionList is not None:
            self.setTransitionList(transitionList)

        if birthDeathList is not None:
            self.setBirthDeathList(birthDeathList)

        if odeList is not None:
            # we have a set of ode explicitly defined!
            if len(odeList) > 0:
                # tests on validity of using odeList
                if transitionList is not None:
                    raise InputError("Transition equations detected even though "
                                     +"the set of ode is explicitly defined")
                if birthDeathList is not None:
                    raise InputError("Birth Death equations detected even though "
                                     +"the set of ode is explicitly defined")

                # set equations
                self.setOdeEquationList(odeList)
            else:
                pass

    ########################################################################
    #
    # Getting and setters
    #
    ########################################################################

    def setParameters(self, parameters):
        '''
        Set the values for the parameters already defined.  Note that unless
        the parameters are entered via a dictionary or a two element list,tuple
        we assume that it is in the order of :meth:`.getParamList`

        Parameters
        ----------
        parameters: list or equivalent
            A list which contains two elements (string, numeric value) or
            just a single array like object

        '''

        # A stupid and complicated type checking procedure.  Someone please kill me
        # when you read this.
        if parameters is not None:
            # currently only accept 3 main types here, obviously apart from the
            # dict type below
            if isinstance(parameters, (list, tuple, numpy.ndarray)):
                # length checking, we are assuming here that we always set the full
                # set of parameters
                if len(parameters) == self._numParam:
                    if type(parameters) is numpy.ndarray:
                        if parameters.size == self._numParam:
                            parameters = parameters.ravel()
                        else:
                            raise InputError("The number of input parameters is "
                                             +str(parameters.size)+ " but "
                                             +str(self._numParam)+  " expected")

                    paramOut = dict()
                else:
                    raise InputError("The number of input parameters is "
                                     +str(len(parameters))+ " but "
                                     +str(self._numParam)+  " expected")

                # type checking, making sure that all the different type are accepted
                if isinstance(parameters[0], tuple):
                    if len(parameters) == self._numParam:
                        for i in range(0, len(parameters)):
                            indexTemp = self._extractParamSymbol(parameters[i][0])
                            valueTemp = parameters[i][1]
                            paramOut[indexTemp] = valueTemp
                # we are happy... I guess
                elif ode_utils.isNumeric(parameters[0]):
                    for i in range(0, len(parameters)):
                        paramOut[self._paramList[i]] = parameters[i]
                else:
                    raise InputError("Input type should either be a list of tuple with "
                                     +"elements (str,numeric) or a list of numeric value")
            elif isinstance(parameters, dict):
                # we assume that the key of the dictionary is a string and
                # the value can be a single value or a distribution
                if len(parameters) > self._numParam:
                    raise Exception("Too many input parameters")

                # holder
                # TODO: change this properly so that there are two different types of parameter
                # input.  One is when we initialize and another when we set new ones
                if self._parameters is None:
                    paramOut = dict()
                else:
                    paramOut = self._parameters

                # extra the key from the parameters dictionary
                for inParam in parameters:
                    value = parameters[inParam]
                    if ode_utils.isNumeric(value):
                        # get index
                        if isinstance(inParam, sympy.Symbol):
                            paramOut[self._extractParamSymbol(str(inParam))] = value
                        else:
                            paramOut[self._extractParamSymbol(inParam)] = value
                        # and replace only that specific one
                    elif isinstance(value, scipy.stats._distn_infrastructure.rv_frozen):
                        # we always assume that we have a frozen distribution
                        paramOut[self._extractParamSymbol(inParam)] = value.rvs(1)[0]
                        # output of the rv from a frozen distribution is a numpy.ndarray even when
                        # the number of sample is one
                        ## Now we are going make damn sure to record it down!
                        self._stochasticParam = parameters
                    elif isinstance(value, tuple):
                        if callable(value[0]):
                            # using a temporary variable to shorten the line.
                            if isinstance(value[1], dict):
                                paramTemp = value[0](1, **value[1])
                            else:
                                paramTemp = value[0](1, *value[1])

                            paramOut[self._extractParamSymbol(inParam)] = paramTemp
                            self._stochasticParam = parameters
                        else:
                            raise InputError("First element should be a callable when using multi "
                                             +"argument distribution definition.  "
                                             +"Type of input was " +str(type(value)))
                    else:
                        raise InputError("Not supported input type for dict() input yet. "
                                         +str(type(value)))
            elif self._numParam == 1:
                # a single parameter ode and you are not evaluating it analytically!
                # fair enough! no further comment your honour.
                paramOut = list()
                if isinstance(parameters, tuple):
                    paramOut[self._extractParamSymbol(parameters[0])] = parameters[1]
                elif isinstance(parameters, (int, float)):
                    paramOut[self.getParamList()[0]] = parameters
                else:
                    raise InputError("Input type should either be a tuple of (str,numeric) "
                                     +"or a single numeric value")
            else:
                raise InputError("Expecting a dict, list or a tuple input because there are a "
                                 +"total of " + str(self._numParam)+ " parameters")
        else:
            if self._numParam != 0:
                raise Warning("Did not set the values of the parameters.  Input was None.")
            else:
                paramOut= dict()

        self._parameters = paramOut

        # unroll the parameter values into the appropriate list
        # if self._paramValue is None or isinstance(self._paramValue, list):
        #     self._paramValue = dict()
        self._paramValue = [0] * len(self._paramList)

        for k, v in self._parameters.iteritems():
            index = self.getParamIndex(k)
            self._paramValue[index] = v

        return self

    def getParameters(self):
        '''
        Returns
        -------
        list
            A list which contains tuple of two elements, (:mod:`sympy.core.symbol`, numeric)

        '''
        return self._paramters

    def setState(self, state):
        '''
        Set the current value for the states and match it to the corresponding symbol

        Parameters
        ----------
        state: tuple
            tuple of two elements, (string, numeric)

        '''
        if state is not None:
            if isinstance(state, (list, tuple)):
                if isinstance(state[0], tuple):
                    stateOut = list()
                    for i in range(0, len(state)):
                        stateOut.append((self._extractStateSymbol(state[i][0]), state[i][1]))

                    self._state = stateOut
                else:
                    self._state = self._unrollState(state)
            elif isinstance(state, numpy.ndarray):
                self._state = self._unrollState(state)
            elif ode_utils.isNumeric(state):
                self._state = self._unrollState(state)
            else:
                raise InputError("Input state is of an unexpected type - "
                                 +type(state))
        else:
            raise InputError("Input state is of an unexpected type - "
                             +type(state))

        return self

    def getState(self):
        '''
        Returns
        -------
        list
            state in symbol with current value, (:mod:`sympy.core.symbol`,numeric)

        '''
        return self._state

    # beware of the fact that
    # time = numeric
    # t = sympy symbol
    def setTime(self, time):
        '''
        Set the time for the ode system

        Parameters
        ----------
        time: numeric
            Current time of the ode

        '''
        if time is not None:
            self._time = time
        return self

    def getTime(self):
        '''
        The current time in the ode system

        Returns
        -------
        numeric

        '''
        return self._time

    def setStateList(self, stateList):
        '''
        Set the set of states for the ode system

        Parameters
        ----------
        stateList: list
            list of string, each string is the name of the state

        '''
        if isinstance(stateList, (list, tuple)):
            for i in range(0, len(stateList)):
                self._addStateSymbol(stateList[i])
        else:
            raise InputError("Expecting a list")
        
        self._hasNewTransition = True
        return self

    def getStateList(self):
        '''
        Returns a list of the states in symbol

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        '''
        return self._stateList
    
    def getParamList(self):
        '''
        Returns a list of the parameters in symbol

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        '''
        return self._paramList
    
    def setParamList(self, paramList):
        '''
        Set the set of parameters for the ode system

        Parameters
        ----------
        paramList: list
            list of string, each string is the name of the parameter

        '''
        if isinstance(paramList, (list, tuple)):
            for i in range(0, len(paramList)):
                self._addParamSymbol(paramList[i])
        else:
            raise InputError("Expecting a list")
        
        self._hasNewTransition = True
        return self
    
    def getDerivedParamList(self):
        '''
        Returns a list of the derived parameters in symbol

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        '''
        return self._derivedParamList
    
    def setDerivedParamList(self, derivedParamList):
        '''
        Set the set of derived parameters for the ode system

        Parameters
        ----------
        derivedParamList: list
            list of string, each string is the name of the derived parameter
            which uses the original parameter
        '''
        for i in range(0, len(derivedParamList)):
            self._addDerivedParam(derivedParamList[i][0], derivedParamList[i][1])

        return self

    # also need to make it transitionScript class
    def setTransitionList(self, transitionList):
        '''
        Set the set of transitions for the ode system

        Parameters
        ----------
        transitionList: list
            list of :class:`.Transition` of type transition in
            :class:`.getTransitionType`
        '''
        if isinstance(transitionList, (list, tuple)):
            for i in range(0, len(transitionList)):
                self.addTransition(transitionList[i])
        else:
            raise InputError("Expecting a list")

        return self

    def getTransitionList(self):
        '''
        Returns a list of the transitions

        Returns
        -------
        list
            with elements as :class:`.Transition`

        '''
        if self._explicitOde == False:
            return self._transitionList
        else:
            raise OutputError("ode was defined explicitly, no transition available")
        
    def setBirthDeathList(self, birthDeathList):
        '''
        Set the set of transitions for the ode system

        Parameters
        ----------
        birthDeathList: list
            list of :class:`.Transition` of type birth or death in
            :class:`.getTransitionType`

        '''
        if isinstance(birthDeathList, (list, tuple)):
            for i in range(0, len(birthDeathList)):
                self.addBirthDeath(birthDeathList[i])
        elif isinstance(birthDeathList, Transition):
            self.addBirthDeath(birthDeathList)
        else:
            raise InputError("Input not as expected.  It is not a list "
                            + "or a Transition")

        return self

    def getBirthDeathList(self):
        '''
        Returns a list of the birth or death process

        Returns
        -------
        list
            with elements as :class:`.Transition`

        '''
        if self._explicitOde == False:
            return self._birthDeathList
        else:
            raise OutputError("ode was defined explicitly, "+
                             "no birth or death process available")

    def setOdeEquationList(self, odeList):
        '''
        Set the set of ode

        Parameters
        ----------
        odeList: list
            list of :class:`.Transition` of type birth or death in
            :class:`.getTransitionType`

        '''
        if isinstance(odeList, list):
            for i in range(0, len(odeList)):
                self.addOdeEquation(odeList[i])
        elif isinstance(odeList, Transition):
            # if it is not a list, then at least it should be an object
            # of the correct type
            self.addOdeEquation(odeList)
        else:
            raise InputError("Input not as expected.  It is not a list "+
                             "or a Transition")

        return self
    
    def getOdeList(self):
        '''
        Returns a list of the ode

        Returns
        -------
        list
            with elements as :class:`.Transition`

        '''
        if self._explicitOde == True:
            return self._odeList
        else:
            raise OutputError("ode was not defined explicitly")

    def getNumState(self):
        '''
        Returns the number of state

        Returns
        -------
        int
            the number of states

        '''
        return len(self._stateList)

    def getNumParam(self):
        '''
        Returns the number of parameters

        Returns
        -------
        int
            the number of parameters

        '''
        return len(self._paramList)

    def getStateIndex(self, inputStr):
        '''
        Finds the index of the state

        Returns
        -------
        int
            the index of the desired state

        '''
        if isinstance(inputStr, str):
            return self._extractStateIndex(inputStr)
        elif isinstance(inputStr, sympy.Symbol):
            return self._extractStateIndex(str(inputStr))
        elif isinstance(inputStr, (list, tuple)):
            outStr = [self._extractStateIndex(stateName) for stateName in inputStr]
#             outStr = list()
#             for stateName in inputStr:
#                 outStr.append(self._extractStateIndex(stateName))

            return outStr

    def getParamIndex(self, inputStr):
        '''
        Finds the index of the parameter

        Returns
        -------
        int
            the index of the desired parameter
        '''
        if isinstance(inputStr, str):
            return self._extractParamIndex(inputStr)
        elif isinstance(inputStr, sympy.Symbol):
            return self._extractParamIndex(str(inputStr))
        elif isinstance(inputStr, (list, tuple)):
            outStr = [self._extractParamIndex(paramName) for paramName in inputStr]
#             outStr = list()
#             for stateName in inputStr:
#                 outStr.append(self._extractParamIndex(stateName))

            return outStr

    ########################################################################
    #
    # Setting the scene
    #
    ########################################################################

    ####
    #
    # The first four is deemed to be "private" to encourage the end user to
    # define things correctly rather than hacking it later on after the
    # ode object has been initialized
    #
    ####
    def _addSymbol(self, inputStr):
        # empty holder to make things look nicer
        x = None
        listSymbol = None
        # now the real code
        if isinstance(inputStr, (list, tuple)):
            if len(inputStr) == 2:
                if str(inputStr[1]).lower() in ("real","complex","true","false"):
                    isReal = 'False'
                elif str(inputStr[1]).lower() in ("real","true"):
                    isReal = 'True'
                else:
                    raise InputError("Unexpected second argument for symbol")
            else:
                raise InputError("Unexpected number of argument for symbol")
        elif isinstance(inputStr, str):
            isReal = 'True'
        else:
            raise InputError("Unexpected input type for symbol")
        
        strAdd = 'x = symbols("' +inputStr+ '", real='+isReal+')'
        exec(strAdd) # this gives us x, our intermediate
        if isinstance(x, sympy.Symbol):
            #strAdd = self._assignToSelf(inputStr)+ ' = Symbol("' +inputStr+ '")'
            strAdd = self._assignToSelf(inputStr)+ ' = symbols("' +inputStr+ '", real='+isReal+')'
            exec(strAdd)
            return eval('self._'+inputStr)
        elif isinstance(x, tuple):
            # tests
            if len(x) == 0:
                # then we have failed already
                raise InputError("Input symbol is not valid")

            if re.search('-', inputStr) is not None:
                raise InputError("Minus sign not allowed in symbol definition")

            # extract the name of the symbol
            symbolStr = re.search('[A-Za-z]+', inputStr).group()
            strAdd = self._assignToSelf(symbolStr)+ ' = symbols("' +inputStr+ '", real='+isReal+')'
            exec(strAdd)
            strDict = 'self._vectorStateDict["' +symbolStr+ '"] = self._'+symbolStr
            # print strDict
            exec(strDict)
            # unroll the symbols from the vector into our internal list
            exec('listSymbol = [i for i in self._' +symbolStr+ ']')
            return listSymbol

    def _addStateSymbol(self, inputStr):
        symbolName = self._addSymbol(inputStr)

        if isinstance(symbolName, sympy.Symbol):
            self._stateList += [symbolName]
            self._stateDict[inputStr] = symbolName
            self._numState += 1
        else:
            self._stateList += symbolName
            for i in symbolName:
                self._stateDict[str(i)] = i
                # we are hacking it here.  Basically, we have unrolled the series of
                # numbered symbol and we wish to also make that available when
                # defining the equations
                symbolName = self._addSymbol(str(i))
                self._numState += 1

        return None

    def _addParamSymbol(self, inputStr):
        symbolName = self._addSymbol(inputStr)

        if isinstance(symbolName, sympy.Symbol):
            if symbolName not in self._paramList:
                self._paramList += [symbolName]
                self._paramDict[inputStr] = symbolName
                self._numParam += 1
        else:
            self._paramList += symbolName
            for s in symbolName:
                if s not in self._paramList:
                    self._paramDict[str(s)] = s
                    # we are hacking it here.  Basically, we have unrolled the series of
                    # numbered symbol and we wish to also make that available when
                    # defining the equations
                    symbolName = self._addSymbol(str(s))
                    self._numParam += 1

        return None

    def _addDerivedParam(self, name, eqn):
        fixedEqn = self._checkEquation(eqn)
        strAdd = self._assignToSelf(name)+ ' = ' +fixedEqn
        exec(strAdd)
        self._derivedParamList.append(eval('self._' +name))
        self._derivedParamDict[name] = eval('self._'+name)
        self._numDerivedParam += 1
        self._hasNewTransition = True
        return None

    def addTransition(self, transition):
        '''
        Add a single transition between two states

        Parameters
        ----------
        transition: :class:`.Transition`
            The transition object that contains all the information
            regarding the transition
        '''
        if isinstance(transition, Transition):
            if transition.getTransitionType() is TransitionType.T:
                self._checkEquation(transition.getEquation())
                self._transitionList.append(transition)
                self._hasNewTransition = True
            else:
                raise InputError("Input is not a transition between two states")
        else:
            raise InputError("Input " +str(type(transition))+ " is not a Transition.")

        return None

    def _computeTransitionMatrix(self):
        '''
        Computes the transition matrix given the transitions
        '''
        # holders
        self._transitionMatrix = sympy.zeros(self._numState, self._numState)

        A = self._transitionMatrix
        # going through the list of transition
        for transition in self._transitionList:
            # then find out the indices of the states
            fromIndex = self._extractStateIndex(transition.getOrigState())
            toIndex = self._extractStateIndex(transition.getDestState())

            # put the getEquation in the correct element
            A[fromIndex,toIndex] += eval(self._checkEquation(transition.getEquation()))
        # assign back
        self._transitionMatrix = A

        return None
    
    def addBirthDeath(self, birthDeath):
        '''
        Add a single birth or death process

        Parameters
        ----------
        transition: :class:`.Transition`
            The transition object that contains all the information
            regarding the process

        '''
        if isinstance(birthDeath, Transition):
            t = birthDeath.getTransitionType()
            if t is TransitionType.B or t is TransitionType.D:
                self._checkEquation(birthDeath.getEquation())
                self._birthDeathList.append(birthDeath)
                self._hasNewTransition = True
            else:
                raise InputError("Input is not a birth death process")
        else:
            raise InputError("Input type is not a Transition")

        return None

    def _computeBirthDeathVector(self):
        # holder
        self._birthDeathVector = zeros(self._numState, 1)
        A = self._birthDeathVector
        # go through all the transition objects
        for i in range(0, len(self._birthDeathList)):
            # extract object
            birthDeath = self._birthDeathList[i]
            # find the (index) state it is related to
            fromIndex = self._extractStateIndex(birthDeath.getOrigState())
            if birthDeath.getTransitionType() is TransitionType.B:
                # birth type, positive sign
                A[fromIndex] += eval(self._checkEquation(birthDeath.getEquation()))
            elif birthDeath.getTransitionType() is TransitionType.D:
                # a death type, negative sign
                A[fromIndex] -= eval(self._checkEquation(birthDeath.getEquation()))

        self._birthDeathVector = A

        return None

    def addOdeEquation(self, eqn):
        '''
        Add an ode

        Parameters
        ----------
        eqn: :class:`.Transition`
            The transition object that contains all the information
            regarding the ode
        '''
        # TODO: check whether previous ode for the same state exist
        # determine if the input object is of the correct type
        if isinstance(eqn, Transition):
            # then whether it is actually an ode
            if eqn.getTransitionType() is TransitionType.ODE:
                # YES!!!
                self._explicitOde = True
                # add to the list
                self._odeList.append(eqn)
            else:
                raise InputError("Input is not a transition of an ode")
        else:
            raise InputError("Input type is not a Transition")

        return None

    def _computeOdeVector(self):
        # we are only testing it here because we want to be flexible and
        # allow the end user to input more state than initially desired
        if len(self._odeList) <= self._numState:
            self._ode = zeros(self._numState, 1)
            A = self._ode
            for i in range(0, len(self._odeList)):
                Transition = self._odeList[i]
                fromIndex = self._extractStateIndex(Transition.getOrigState())
                A[fromIndex] = eval(self._checkEquation(Transition.getEquation()))

            self._ode = A
        else:
            raise InputError("The total number of ode is "+str(len(self._odeList))+
                            " where the number of state is "+str(self._numState))

        return None

    ########################################################################
    # Unrolling of the information
    # state
    ########################################################################

    def _unrollState(self, state):
        '''
        Information unrolling from vector to sympy in state
        '''
        stateOut = list()
        if self._stateList == 1:
            if ode_utils.isNumeric(state):
                stateOut.append((self._stateList[0], state))
            else:
                raise InputError("Number of input state not as expected")
        else:
            if len(state) == len(self._stateList):
                for i in range(0, self._numState):
                    stateOut.append((self._stateList[i], state[i]))
            else:
                raise InputError("Number of input state not as expected")

        return stateOut

    ########################################################################
    #
    # Ugly shit that is required to fix strings to sympy symbols
    #
    ########################################################################

    def _checkEquation(self, inputStr):
        for strParam in self._paramDict:
            inputStr = self._substituteSelf(inputStr, strParam)

        for strState in self._stateDict:
            inputStr = self._substituteSelf(inputStr, strState)

        for strVectorState in self._vectorStateDict:
            inputStr = self._substituteSelf(inputStr, strVectorState)

        for strDerivedParam in self._derivedParamDict:
            inputStr = self._substituteSelf(inputStr, strDerivedParam)

        # if the evaluation fails then there is a problem with the
        # variables
        #print inputStr
        eval(inputStr)
        return inputStr

    def _simplifyEquation(self, inputStr):
        sList = list()
        # these are considered the "dangerous" operation that will
        # overflow in numpy
        sList.append(len(inputStr.atoms(exp)))
        sList.append(len(inputStr.atoms(log)))

        if numpy.sum(sList) != 0:
        # it is dangerous to simplify!
            self._isDifficult = True
            return inputStr
        else:
            return simplify(inputStr)

    def _substituteSelf(self, inputStr, inputName):
        '''
        given a string, we want to substitute part of the string
        that matches "inputName" to one that has "self._" in front
        of it.  This then allows the string, getEquation in this case,
        to be evaluated later because all the states and parameters
        have already been mapped to something internal, i.e. with
        a prefix of "self._"
        '''
        strNew = 'r\'' +self._assignToSelf(inputName)+ '\''
        strTarget = 'r\'' + r'\b' +inputName+ r'\b'+ '\''
        inputStr = re.sub(eval(strTarget), eval(strNew), inputStr)
        return (inputStr)

    def _assignToSelf(self, inputStr):
        return 'self._'+inputStr

    def _stateExist(self, inputStr):
        return self._stateDict.has_key(inputStr)

    def _paramExist(self, inputStr):
        return self._paramDict.has_key(inputStr)

    def _derivedParamExist(self, inputStr):
        return self._derivedParamDict.has_key(inputStr)

    def _extractParamIndex(self, inputStr):
        if self._paramExist(inputStr):
            return self._paramList.index(self._paramDict[inputStr])
        else:
            raise InputError("Input parameter: "+inputStr+ " does not exist")

    def _extractParamSymbol(self, inputStr):
        if self._paramExist(inputStr):
            return self._paramDict[inputStr]
        else:
            raise InputError("Input parameter: "+inputStr+ " does not exist")

    def _extractStateIndex(self, inputStr):
        if self._stateExist(inputStr):
            return self._stateList.index(self._stateDict[inputStr])
        elif self._stateExist(str(eval(self._assignToSelf(inputStr)))):
            # need the `eval` command to get the symbol out of the vector
            # then the `str` to convert to a string that is recognizable by the dict
            return self._stateList.index(self._stateDict[str(eval(self._assignToSelf(inputStr)))])
        else:
            raise Exception("Input state: "+inputStr+ " does not exist")

    def _extractStateSymbol(self, inputStr):
        if self._stateExist(inputStr):
            return self._stateDict[inputStr]
        else:
            raise InputError("Input state: "+inputStr+ " does not exist")

    def _extractUpperTriangle(self, A, nrow=None, ncol=None):
        '''
        Extract the upper triangle of matrix A

        Parameters
        ----------
        A: :mod:`sympy.matrices.matrices`
            input matrix
        nrow: int
            number of row
        ncol: int
            number of column

        Returns
        -------
        :mod:`sympy.matrices.matrices`
            An upper triangle matrix

        '''
        if nrow is None:
            nrow = len(A[:,0])

        if ncol is None:
            ncol = len(A[0,:])

        B = zeros(nrow, ncol)
        for i in range(0, nrow):
            for j in range(i, ncol):
                B[i,j] = A[i,j]

        return B
