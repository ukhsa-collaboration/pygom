"""

    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    This module contains the classes required to translate inputs in string
    into an algebraic machine using sympy

"""
# string evaluation
import re
reMath = re.compile(r'[-+*\\]')
reUnderscore = re.compile('^_')
reSymbolName = re.compile('[A-Za-z_]+')
reSymbolIndex = re.compile('.*\[([0-9]+)\]$')
reSplitString = re.compile(',|\s')

import sympy
from sympy import symbols
# numerical computation
import numpy
from scipy.stats._distn_infrastructure import rv_frozen

from .transition import Transition, TransitionType
from ._model_errors import InputError, OutputError
from ._model_verification import checkEquation
from .ode_variable import ODEVariable
from . import ode_utils

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
        self._derivedParamEqn = list()

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
        
        # information about the ode in general
        # reactant, state change and the dependency graph matrix
        self._lambdaMat = None
        self._vMat = None
        self._GMat = None

        if stateList is not None:
            if isinstance(stateList, str):
                stateList = reSplitString.split(stateList)
                stateList = filter(lambda x: not len(x.strip()) == 0, stateList)
            self.setStateList(list(stateList))

        if paramList is not None:
            if isinstance(paramList, str):
                paramList = reSplitString.split(paramList)
                paramList = filter(lambda x: not len(x.strip()) == 0, paramList)
            self.setParamList(list(paramList))

        # this has to go after adding the parameters
        # because it is suppose to be based on the current
        # base parameters.
        # Making the distinction here because it makes a
        # difference when inferring the parameters of the variables
        if not ode_utils._noneOrEmptyList(derivedParamList):
            self.setDerivedParamList(derivedParamList)
        # if derivedParamList is not None:

        # if transitionList is not None:
        if not ode_utils._noneOrEmptyList(transitionList):
            self.setTransitionList(transitionList)

        # if birthDeathList is not None:
        if not ode_utils._noneOrEmptyList(birthDeathList):
            self.setBirthDeathList(birthDeathList)

        # if odeList is not None:
        if not ode_utils._noneOrEmptyList(odeList):
            # we have a set of ode explicitly defined!
            if len(odeList) > 0:
                # tests on validity of using odeList
                # if transitionList is not None:
                if not ode_utils._noneOrEmptyList(transitionList):
                    raise InputError("Transition equations detected even " +
                                     "though the set of ode is explicitly " + 
                                     "defined")
                # if birthDeathList is not None:
                if not ode_utils._noneOrEmptyList(birthDeathList):
                    raise InputError("Birth Death equations detected even " + 
                                     "though the set of ode is explicitly " + 
                                     "defined")

                # set equations
                self.setOdeEquationList(odeList)
            else:
                pass
            
        self.getNumTransitions()
        self._transitionVector = self._computeTransitionVector()

    def _getModelStr(self):
        modelStr = "(%s, %s, %s, %s, %s, %s)" % (self._stateList, 
                                                 self._paramList,
                                                 self._derivedParamEqn,
                                                 self._transitionList, 
                                                 self._birthDeathList,
                                                 self._odeList)
        if self._parameters is not None:
            modelStr += ".setParameters(%s)" % \
                        {str(k): v for k, v in self._parameters.items()}
        return(modelStr)

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
        
        err_string = "The number of input parameters is %s but %s expected"
        # setting up a shorthand for the most used function within this method
        f = self._extractParamSymbol
        # A stupid and complicated type checking procedure.  Someone please
        # kill me when you read this.
        if parameters is not None:
            # currently only accept 3 main types here, obviously apart
            # from the dict type below
            if isinstance(parameters, (list, tuple, numpy.ndarray)):
                # length checking, we are assuming here that we always set
                # the full set of parameters
                if len(parameters) == self._numParam:
                    if isinstance(parameters, numpy.ndarray):
                        if parameters.size == self._numParam:
                            parameters = parameters.ravel()
                        else:
                            raise InputError(err_string % \
                                             (parameters.size, self._numParam))

                    paramOut = dict()
                else:
                    raise InputError(err_string % \
                                     (parameters.size, self._numParam))

                # type checking, making sure that all the different types
                # are accepted
                if isinstance(parameters[0], tuple):
                    if len(parameters) == self._numParam:
                        for i in range(0, len(parameters)):
                            indexTemp = f(parameters[i][0])
                            valueTemp = parameters[i][1]
                            paramOut[indexTemp] = valueTemp
                # we are happy... I guess
                elif ode_utils.isNumeric(parameters[0]):
                    for i in range(len(parameters)):
                        if isinstance(self._paramList[i], ODEVariable):
                            paramOut[str(self._paramList[i])] = parameters[i]
                        else:
                            paramOut[self._paramList[i]] = parameters[i]
                else:
                    raise InputError("Input type should either be a list of " +
                                     "tuple with elements (str,numeric) or " +
                                     "a list of numeric value")
            elif isinstance(parameters, dict):
                # we assume that the key of the dictionary is a string and
                # the value can be a single value or a distribution
                if len(parameters) > self._numParam:
                    raise Exception("Too many input parameters")

                # holder
                # TODO: change this properly so that there are two different
                # types of parameter input.  One is when we initialize and
                # another when we set new ones
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
                            paramOut[f(str(inParam))] = value
                        else:
                            paramOut[f(inParam)] = value
                        # and replace only that specific one
                    elif isinstance(value, rv_frozen):
                        # we always assume that we have a frozen distribution
                        paramOut[f(inParam)] = value.rvs(1)[0]
                        # output of the rv from a frozen distribution is a
                        # numpy.ndarray even when the number of sample is one
                        ## Now we are going make damn sure to record it down!
                        self._stochasticParam = parameters
                    elif isinstance(value, tuple):
                        if callable(value[0]):
                            # using a temporary variable to shorten the line.
                            if isinstance(value[1], dict):
                                paramTemp = value[0](1, **value[1])
                            else:
                                paramTemp = value[0](1, *value[1])

                            paramOut[f(inParam)] = paramTemp
                            self._stochasticParam = parameters
                        else:
                            raise InputError("First element should be a " +
                                             "callable when using multi " +
                                             "argument distribution " + 
                                             "definition.  Type of input " +
                                             "was " + str(type(value)))
                    else:
                        raise InputError("No supported input type " + 
                                         str(type(value)) + " for " +
                                         "dict() input yet.")
            elif self._numParam == 1:
                # a single parameter ode and you are not evaluating it
                # analytically! fair enough! no further comments your honour.
                paramOut = list()
                if isinstance(parameters, tuple):
                    paramOut[f(parameters[0])] = parameters[1]
                elif isinstance(parameters, (int, float)):
                    paramOut[self.getParamList()[0]] = parameters
                else:
                    raise InputError("Input type should either be a tuple of " +
                                     "(str,numeric) or a single numeric value")
            else:
                raise InputError("Expecting a dict, list or a tuple input " + 
                                 "because there are a total of " + 
                                 str(self._numParam)+ " parameters")
        else:
            if self._numParam != 0:
                raise Warning("Did not set the values of the parameters. " +
                              "Input was None.")
            else:
                paramOut= dict()

        self._parameters = paramOut

        # unroll the parameter values into the appropriate list
        # if self._paramValue is None or isinstance(self._paramValue, list):
        #     self._paramValue = dict()
        self._paramValue = [0]*len(self._paramList)

        for k, v in self._parameters.items():
            index = self.getParamIndex(k)
            self._paramValue[index] = v

        return(self)

    def getParameters(self):
        '''
        Returns
        -------
        list
            A list which contains tuple of two elements,
            (:mod:`sympy.core.symbol`, numeric)

        '''
        return(self._parameters)

    def setStateValue(self, state):
        '''
        Set the current value for the states and match it to the
        corresponding symbol

        Parameters
        ----------
        state: array like
            either a vector of numeric value of a
            tuple of two elements, (string, numeric)

        '''
        err_str = "Input state is of an unexpected type - " + type(state)

        if state is not None:
            if isinstance(state, (list, tuple)):
                if isinstance(state[0], tuple):
                    self._state = []
                    for s in state:
                        self._state += [self._extractStateSymbol(s[0], s[1])]
                else:
                    self._state = self._unrollState(state)
            elif isinstance(state, numpy.ndarray):
                self._state = self._unrollState(state)
            elif ode_utils.isNumeric(state):
                self._state = self._unrollState(state)
            else:
                raise InputError(err_str)
        else:
            raise InputError(err_str)

        return(self)

    def getState(self):
        '''
        Returns
        -------
        list
            state in symbol with current value, 
            (:mod:`sympy.core.symbol`,numeric)

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
        return(self)

    def getTime(self):
        '''
        The current time in the ode system

        Returns
        -------
        numeric

        '''
        return(self._time)

    def setStateList(self, stateList):
        '''
        Set the set of states for the ode system

        Parameters
        ----------
        stateList: list
            list of string, each string is the name of the state

        '''
        if isinstance(stateList, (list, tuple)):
            for s in stateList:
                self._addStateSymbol(s)
        elif isinstance(stateList, (str, ODEVariable)):
            self._addStateSymbol(stateList)
        else:
            raise InputError("Expecting a list")
        
        self._hasNewTransition = True
        return(self)

    def getStateList(self):
        '''
        Returns a list of the states in symbol

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        '''
        return(self._stateList)

    def getParamList(self):
        '''
        Returns a list of the parameters in symbol

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        '''
        return(self._paramList)

    def setParamList(self, paramList):
        '''
        Set the set of parameters for the ode system

        Parameters
        ----------
        paramList: list
            list of string, each string is the name of the parameter

        '''
        if isinstance(paramList, (list, tuple)):
            for p in paramList:
                self._addParamSymbol(p)
        elif isinstance(paramList, (str, ODEVariable)):
            self._addParamSymbol(paramList)
        else:
            raise InputError("Expecting a list")
        
        self._hasNewTransition = True
        return(self)

    def getDerivedParamList(self):
        '''
        Returns a list of the derived parameters in symbol

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        '''
        return(self._derivedParamList)

    def setDerivedParamList(self, derivedParamList):
        '''
        Set the set of derived parameters for the ode system

        Parameters
        ----------
        derivedParamList: list
            list of string, each string is the name of the derived parameter
            which uses the original parameter
        '''
        for param in derivedParamList:
            self._addDerivedParam(param[0], param[1])

        return(self)

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
            for i in range(len(transitionList)):
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
            raise OutputError("ode was defined explicitly, no " +
                              "transition available")
        
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
            for i in range(len(birthDeathList)):
                self.addBirthDeath(birthDeathList[i])
        elif isinstance(birthDeathList, Transition):
            self.addBirthDeath(birthDeathList)
        else:
            raise InputError("Input not as expected.  It is not a list " +
                             "or a Transition")

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
            raise OutputError("ode was defined explicitly, " +
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
            for i in range(len(odeList)):
                self.addOdeEquation(odeList[i])
        elif isinstance(odeList, Transition):
            # if it is not a list, then at least it should be an object
            # of the correct type
            self.addOdeEquation(odeList)
        else:
            raise InputError("Input not as expected.  It is not a list " +
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
        if isinstance(inputStr, sympy.Symbol):
            return self._extractStateIndex(str(inputStr))
        elif isinstance(inputStr, ODEVariable):
            return self._extractStateIndex(inputStr.ID)
        else:
            return self._extractStateIndex(inputStr)

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
        elif isinstance(inputStr, ODEVariable):
            return self._extractParamIndex(inputStr.ID)
        elif isinstance(inputStr, (list, tuple)):
            outStr = [self._extractParamIndex(p) for p in inputStr]
            return outStr

    def getNumTransitions(self):
        '''
        Returns the total number of transition objects that belongs to
        either a pure transition or a birth/death process
        
        Returns
        -------
        int
            total number of transitions
        '''
        self._numPureTransition = len(self._transitionList)
        self._numBD = len(self._birthDeathList)
        self._numTransition = self._numPureTransition + self._numBD
        return(self._numTransition)

    ########################################################################
    #
    # Setting the scene
    #
    ########################################################################

    ####
    #
    # Most of the followings are deemed to be "private" to encourage the
    # end user to define things correctly rather than hacking it later
    # on after the ode object has been initialized
    #
    ####
    
    def _addSymbol(self, inputStr):
        assert reMath.search(inputStr) is None, \
            "Mathematical operators not allowed in symbol definition"
        assert reUnderscore.search(inputStr) is None, \
            "A symbol cannot have underscore as first character"

        if isinstance(inputStr, (list, tuple)):
            if len(inputStr) == 2:
                if str(inputStr[1]).lower() in ("complex", "false"):
                    isReal = 'False'
                elif str(inputStr[1]).lower() in ("real","true"):
                    isReal = 'True'
                else:
                    raise InputError("Unexpected second argument for symbol")
            else:
                raise InputError("Unexpected number of argument for symbol")
        elif isinstance(inputStr, str): # assume real unless stated otherwise
            isReal = 'True'
        else:
            raise InputError("Unexpected input type for symbol")

        assert inputStr!='lambda', "lambda is a reserved keyword"
        tempSym = eval("symbols('%s', real=%s)" % (inputStr, isReal))
        
        if isinstance(tempSym, sympy.Symbol):
            return tempSym
        elif isinstance(tempSym, tuple):
            assert len(tempSym) != 0, "Input symbol is not valid"
            # extract the name of the symbol
            symbolStr = reSymbolName.search(inputStr).group()
            self._vectorStateDict[symbolStr] = tempSym
            return list(tempSym)
        else:
            raise InputError("Unexpected result using the input string:"
                             + str(tempSym))

    def _addStateSymbol(self, inputStr):
        if isinstance(inputStr, str):
            varObj = ODEVariable(inputStr, inputStr)
        elif isinstance(inputStr, ODEVariable):
            varObj = inputStr

        symbolName = self._addSymbol(varObj.ID)
  
        if isinstance(symbolName, sympy.Symbol):
            if symbolName not in self._paramList:
                self._numState = self._addVariable(symbolName, varObj, 
                                self._stateList, self._stateDict, 
                                self._numState)
        else:
            for sym in symbolName:
                self._addStateSymbol(str(sym))

        return None

    def _addParamSymbol(self, inputStr):
        if isinstance(inputStr, str):
            varObj = ODEVariable(inputStr, inputStr)
        elif isinstance(inputStr, ODEVariable):
            varObj = inputStr
 
        symbolName = self._addSymbol(varObj.ID)
  
        if isinstance(symbolName, sympy.Symbol):
            if symbolName not in self._paramList:
                self._numParam = self._addVariable(symbolName, varObj, 
                                self._paramList, self._paramDict, 
                                self._numParam)
        else:
            for sym in symbolName:
                self._addParamSymbol(str(sym))
        return None

    def _addDerivedParam(self, name, eqn):
        varObj = ODEVariable(name, name)
        fixedEqn = checkEquation(eqn, *self._getListOfVariablesDict())
        self._numDerivedParam = self._addVariable(fixedEqn, varObj, 
                                self._derivedParamList, self._derivedParamDict, 
                                self._numDerivedParam)
        self._hasNewTransition = True
        self._derivedParamEqn += [(name, eqn)]
        return None

    def _addVariable(self, symbol, varObj, objList, objDict, objCounter):
        assert isinstance(varObj, ODEVariable), "Expecting type odeVariable"
        objList.append(varObj)
        objDict[varObj.ID] = symbol
        return objCounter + 1

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
                self._transitionList.append(transition)
                self._hasNewTransition = True
            else:
                raise InputError("Input is not a transition between two states")
        else:
            raise InputError("Input %s is not a Transition." % type(transition))

        return None

    def _computeTransitionMatrix(self):
        '''
        Computes the transition matrix given the transitions
        '''
        # holders
        self._transitionMatrix = sympy.zeros(self._numState, self._numState)
        # going through the list of transitions
        pure_trans = self._getAllTransition(pureTransitions=True)
        fromList, toList, eqnList = self._unrollTransitionList(pure_trans)
        for k, eqn in enumerate(eqnList):
            for i in fromList[k]:
                for j in toList[k]:
                    self._transitionMatrix[i,j] += eqn

        return self._transitionMatrix
    
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
                self._birthDeathList.append(birthDeath)
                self._hasNewTransition = True
            else:
                raise InputError("Input is not a birth death process")
        else:
            raise InputError("Input type is not a Transition")

        return None

    def _computeBirthDeathVector(self):
        # holder
        self._birthDeathVector = sympy.zeros(self._numState, 1)
        # go through all the transition objects
        for bdObj in self._birthDeathList:
            fromIndex, _toIndex, eqn = self._unrollTransition(bdObj)
            for i in fromIndex:
                if bdObj.getTransitionType() is TransitionType.B:
                    self._birthDeathVector[i] += eqn                
                elif bdObj.getTransitionType() is TransitionType.D:
                    self._birthDeathVector[i] -= eqn

        return self._birthDeathVector

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
            self._ode = sympy.zeros(self._numState, 1)
            fromList, _t, eqnList = self._unrollTransitionList(self._odeList)
            for i, eqn in enumerate(eqnList):
                if len(fromList[i]) > 1:
                    raise InputError("An explicit ode cannot describe more " + 
                                     "than a single state")
                else:
                    self._ode[fromList[i][0]] = eqn
        else:
            raise InputError("The total number of ode is %s " +
                             "where the number of state is %s" % \
                             (len(self._odeList), self._numState))


        return(None)
    
    def _computeTransitionVector(self):
        '''
        Get all the transitions into a vector, arranged by state to
        state transition then the birth death processes
        '''
        self._transitionVector = sympy.zeros(self._numTransition, 1)
        _f, _t, eqnList = self._unrollTransitionList(self._getAllTransition())
        for i, eqn in enumerate(eqnList):
            self._transitionVector[i] = eqn

        return(self._transitionVector)
    
    ########################################################################
    #
    # Other type of matrices
    #
    ########################################################################

    def _computeReactantMatrix(self):
        '''
        The reactant matrix, where

        .. math::
            \lambda_{i,j} = \left\{ 1, &if state i is involved in transition j, \\
                                    0, &otherwise \right.
        '''
        # declare holder
        self._lambdaMat = numpy.zeros((self._numState, self._numTransition), int)

        _fromList, _toList, eqnList = self._unrollTransitionList(self._getAllTransition())
        for j, eqn in enumerate(eqnList):
            for i, state in enumerate(self._stateList):
                if self._stateDict[state.ID] in eqn.atoms():
                    self._lambdaMat[i,j] = 1

        return(self._lambdaMat)

    def _computeStateChangeMatrix(self):
        '''
        The state change matrix, where
        .. math::
            v_{i,j} = \left\{ 1, &if transition j cause state i to lose a particle, \\
                             -1, &if transition j cause state i to gain a particle, \\
                              0, &otherwise \right.
        '''
        self._vMat = numpy.zeros((self._numState, self._numTransition), int)

        fromList, toList, eqnList = self._unrollTransitionList(self._getAllTransition())
        for j, _eqn in enumerate(eqnList):
            if j < self._numPureTransition:
                for k1 in fromList[j]:
                    self._vMat[k1,j] += -1
                for k2 in toList[j]:
                    self._vMat[k2,j] += 1
            else:
                bdObj = self._birthDeathList[j - self._numPureTransition]
                if bdObj.getTransitionType() is TransitionType.B:
                    for k1 in fromList[j]:
                        self._vMat[k1,j] += 1
                elif bdObj.getTransitionType() is TransitionType.D:
                    for k2 in fromList[j]:
                        self._vMat[k2,j] += -1

        return(self._vMat)

    def _computeDependencyMatrix(self):
        '''
        Obtain the dependency matrix/graph. G_{i,j} indicate whether invoking
        the transition j will cause the rate to change for transition j
        '''
        if self._lambdaMat is None:
            self._computeReactantMatrix()
        if self._vMat is None:
            self._computeStateChangeMatrix()

        self._GMat = numpy.zeros((self._numTransition, self._numTransition), int)

        for i in range(self._numTransition):
            for j in range(self._numTransition):
                d = 0
                for k in range(self._numState):
                    d = d or (self._lambdaMat[k,i] and self._vMat[k,j])
                self._GMat[i,j] = d

        return(self._GMat)


    ########################################################################
    # Unrolling of the information
    # state
    ########################################################################

    def _unrollState(self, state):
        '''
        Information unrolling from vector to sympy in state
        '''
        stateOut = list()
        if len(self._stateList) == 1:
            if ode_utils.isNumeric(state):
                stateOut.append((self._stateList[0], state))
            else:
                raise InputError("Number of input state not as expected")
        else:
            if len(state) == len(self._stateList):
                for i in range(self._numState):
                    stateOut.append((self._stateList[i], state[i]))
            else:
                raise InputError("Number of input state not as expected")

        return(stateOut)
    
    def _unrollTransition(self, transitionObj):
        '''
        Given a transition object, get the information from it in a usable
        format i.e. indexing within this class
        '''
        origState = transitionObj.getOrigState()
        fromIndex = self._extractStateIndex(origState)
       
        destState = transitionObj.getDestState()
        toIndex = self._extractStateIndex(destState)

        eqn = checkEquation(transitionObj.getEquation(),
                            *self._getListOfVariablesDict())
        return fromIndex, toIndex, eqn
    
    def _unrollTransitionList(self, transitionList):
        fromList = list()
        toList = list()
        eqnList = list()
        for t in transitionList:
            fromList.append(self._extractStateIndex(t.getOrigState()))
            toList.append(self._extractStateIndex(t.getDestState()))
            eqnList.append(t.getEquation())
        
        eqnList = checkEquation(eqnList, *self._getListOfVariablesDict())
        eqnList = eqnList if hasattr(eqnList, '__iter__') else [eqnList]
        return fromList, toList, eqnList
    
    def _getAllTransition(self, pureTransitions=False):
        
        n = self._numPureTransition if pureTransitions else self._numTransition
        
        allTransitionList = list() 
        for j in range(n):
            if j < self._numPureTransition:
                allTransitionList.append(self._transitionList[j])
            else:
                _i = j - self._numPureTransition
                allTransitionList.append(self._birthDeathList[_i])
        return allTransitionList
    
    def _iterStateList(self):
        '''
        Iterator through the states in symbolic form
        '''
        for s in self._stateList:
            yield self._stateDict[s.ID]

    def _iterParamList(self):
        '''
        Iterator through the parameters in symbolic form
        '''
        for p in self._paramList:
            yield self._paramDict[p.ID]
    
    def _getListOfVariablesDict(self):
        paramDict = [self._paramDict, self._stateDict, self._vectorStateDict] 
        return paramDict, self._derivedParamDict

    ########################################################################
    #
    # Ugly shit that is required to fix strings to sympy symbols
    #
    ########################################################################

    def _extractParamIndex(self, inputStr):
        if inputStr in self._paramDict:
            return self._paramList.index(self._paramDict[inputStr])
        else:
            raise InputError("Input parameter: " +inputStr+ " does not exist")

    def _extractParamSymbol(self, inputStr):
        if isinstance(inputStr, ODEVariable):
            inputStr = inputStr.ID

        if inputStr in self._paramDict:
            return self._paramDict[inputStr]
        else:
            raise InputError("Input parameter: "+inputStr+ " does not exist")

    def _extractStateIndex(self, inputStr):
        if inputStr is None:
            return list()
        else:
            if isinstance(inputStr, (str, sympy.Symbol)):
                inputStr = [inputStr] # make this an iterable
        
            if hasattr(inputStr, '__iter__'):
                return [self._extractStateIndexSingle(i) for i in inputStr]
            else:
                raise Exception("Input must be a string or an iterable " + 
                                "object of string")
            
    def _extractStateIndexSingle(self, inputStr):
        if isinstance(inputStr, ODEVariable):
            return(self._stateList.index(inputStr)) 
        else:
            symName = self._extractStateSymbol(inputStr)
            return(self._stateList.index(symName))

    def _extractStateSymbol(self, inputStr):
        if isinstance(inputStr, ODEVariable):
            inputStr = inputStr.ID

        if inputStr in self._stateDict:
            return(self._stateDict[inputStr])
        else:
            symName = reSymbolName.search(inputStr)
            if symName is not None:
                if symName.group() in self._vectorStateDict:
                    index = reSymbolIndex.findall(inputStr) 
                    if index is not None and len(index) == 1:
                        _i = int(index[0])
                        return self._vectorStateDict[symName.group()][_i]
                    else:
                        raise InputError("Cannot find input state, input %s " + 
                                         "appears to be a vector that was " + 
                                         "not initialized" % symName)
                else:
                    raise InputError("Cannot find input state, input %s " + 
                                     "likely to be a vector" % symName)
            else:
                raise InputError("Input state: " + inputStr + " does not exist")

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

        B = sympy.zeros(nrow, ncol)
        for i in range(0, nrow):
            for j in range(i, ncol):
                B[i,j] = A[i,j]

        return B

