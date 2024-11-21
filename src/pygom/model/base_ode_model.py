"""

    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    This module contains the classes required to translate inputs in string
    into an algebraic machine using sympy

"""
# string evaluation
import re
from numbers import Number

import sympy
import numpy as np
from sympy import symbols
from scipy.stats._distn_infrastructure import rv_frozen

from .transition import Event, Transition, TransitionType
from ._model_errors import InputError, OutputError
from ._model_verification import checkEquation
from .ode_variable import ODEVariable
from . import ode_utils

re_math = re.compile(r'[-+*\\]')
re_underscore = re.compile('^_')
re_symbol_name = re.compile('[A-Za-z_]+')
re_symbol_index = re.compile(r'.*\[([0-9]+)\]$')
re_split_string = re.compile(r',|\s')

class HasNewTransition(ode_utils.CompileCanary):
    states = []

class BaseOdeModel(object):
    """
    This contains the base that stores all the information of an ode

    Parameters
    ----------
    state: list
        A list of states (string)
    param: list
        A list of the parameters (string)
    derived_param: list
        A list of the derived parameters (tuple of (string, string))
    transition: list
        A list of transition (:class:`.Transition`)
    event: list
        A list of events (:class:`.Transition`)
    birth_death: list
        A list of birth or death process (:class:`.Transition`)
    ode: list
        A list of ode (:class:`.Transition`)

    """

    def __init__(self,
                 state=None,
                 param=None,
                 derived_param=None,
                 transition=None,
                 event=None,
                 birth_death=None,
                 ode=None):
        """
        Constructor
        # """

        # TODO: This is probably cluttered with definitions that are unnecessary.
        #       Need to comb through.

        # # the 3 required inputs when doing evaluation
        # self._state = None
        # self._param = None
        # self._time = None
        # self._state_lims=None

        # # we always need time to be a symbol and it should be denoted as t
        self._t = symbols('t')

        self._isDifficult = False

        # # allows the system to be defined directly
        # self._ode = None
        self._odeList = list()
        # self._explicitOde = False

        # # book keeping parameters/states and etc
        self._paramList = list()
        # # holder for the values of the parameters
        self._paramValue = None

        self._stateList = list()
        self._derivedParamList = list()
        self._derivedParamEqn = list()

        # # Derived states:
        # # these differ from params since we know the d/dt but not algebraic forms

        # # this three is not actually that useful
        # # but lets leave it here for now
        # self._parameters = None
        self._stochasticParam = None
        self._hasNewTransition = HasNewTransition()

        # # dictionary for mapping
        self._paramDict = dict()
        # # although time is not defined as a parameter, we want to keep
        # # record of it existence
        self._paramDict['t'] = self._t
        self._stateDict = dict()
        self._derivedParamDict = dict()

        # # dictionary to store vector symbols
        self._vectorStateDict = dict()

        # #  holders for the actual equations
        self._transitionList = list()
        self._eventList = list()
        # self._transitionMatrix = None
        self._birthDeathList = list()
        self._birthDeathVector = list()

        # self.tstep=False

        self._add_list_attr_with_limits(state, "state_list")
        self._add_list_attr(param, "param_list")

        # this has to go after adding the parameters
        # because it is suppose to be based on the current
        # base parameters.
        # Making the distinction here because it makes a
        # difference when inferring the parameters of the variables
        if not ode_utils.none_or_empty_list(derived_param):
            self.derived_param_list = derived_param

        if not ode_utils.none_or_empty_list(event):
            self.event_list = event

        if not ode_utils.none_or_empty_list(transition):
            self.transition_list = transition

        if not ode_utils.none_or_empty_list(birth_death):
            self.birth_death_list = birth_death

        if not ode_utils.none_or_empty_list(ode):
            self.ode_list = ode

        #self._computeEventRateVector()

    ###########################################################################
    #
    # Getters and setters
    #
    ###########################################################################

    @property
    def parameters(self):
        """
        Returns
        -------
        list
            A list which contains tuple of two elements,
            (:mod:`sympy.core.symbol`, numeric)

        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        """
        Set the values for the parameters already defined.  Note that unless
        the parameters are entered via a dictionary or a two element list,tuple
        we assume that it is in the order of :meth:`.getParamList`

        Parameters
        ----------
        parameters: list or equivalent
            A list which contains two elements (string, numeric value) or
            just a single array like object

        """

        err_string = "The number of input parameters is %s but %s expected"
        # setting up a shorthand for the most used function within this method
        f = self._extractParamSymbol
        # A stupid and complicated type checking procedure.  Someone please
        # kill me when you read this.
        # TODO: Would be good to clean this up.
        param_out = dict()
        if parameters is not None:
            # currently only accept 3 main types here, obviously apart
            # from the dict type below
            if isinstance(parameters, (list, tuple, np.ndarray)):
                # length checking, we are assuming here that we always set
                # the full set of parameters
                # TODO: for model fitting, we might only want to set a subset of the known ones
                if len(parameters) == self.num_param:
                    if isinstance(parameters, np.ndarray):
                        if parameters.size == self.num_param:
                            parameters = parameters.ravel()
                        else:
                            raise InputError(err_string % \
                                             (parameters.size, self.num_param))
                else:
                    raise InputError(err_string % \
                                     (parameters.size, self.num_param))

                # type checking, making sure that all the different types
                # are accepted
                if isinstance(parameters[0], tuple):
                    if len(parameters) == self.num_param:
                        for i in range(0, len(parameters)):
                            index_temp = f(parameters[i][0])
                            value_temp = parameters[i][1]
                            param_out[index_temp] = value_temp
                # we are happy... I guess
                elif isinstance(parameters[0], Number):
                    for i, pi in enumerate(parameters):
                        if isinstance(self._paramList[i], ODEVariable):
                            param_out[str(self._paramList[i])] = pi
                        else:
                            param_out[self._paramList[i]] = pi
                else:
                    raise InputError("Input type should either be a list of " +
                                     "tuple with elements (str,numeric) or " +
                                     "a list of numeric value")
            elif isinstance(parameters, dict):
                # we assume that the key of the dictionary is a string and
                # the value can be a single value or a distribution
                if len(parameters) > self.num_param:
                    raise Exception("Too many input parameters")

                # holder
                # TODO: change this properly so that there are two different
                # types of parameter input.  One is when we initialize and
                # another when we set new ones
                if hasattr(self, "_parameters"):
                    param_out = self._parameters

                # extra the key from the parameters dictionary
                for inParam in parameters:
                    value = parameters[inParam]
                    if isinstance(value, Number):
                        # get index
                        if isinstance(inParam, sympy.Symbol):
                            param_out[f(str(inParam))] = value
                        else:
                            param_out[f(inParam)] = value
                        # and replace only that specific one
                    elif isinstance(value, rv_frozen):
                        # we always assume that we have a frozen distribution
                        param_out[f(inParam)] = value.rvs(1)[0]
                        # output of the rv from a frozen distribution is a
                        # np.ndarray even when the number of sample is one
                        ## Now we are going make damn sure to record it down!
                        self._stochasticParam = parameters
                    elif isinstance(value, tuple):
                        if callable(value[0]):
                            # using a temporary variable to shorten the line.
                            if isinstance(value[1], dict):
                                paramTemp = value[0](1, **value[1])
                            else:
                                paramTemp = value[0](1, *value[1])

                            param_out[f(inParam)] = paramTemp
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
            elif self.num_param == 1:
                # a single parameter ode and you are not evaluating it
                # analytically! fair enough! no further comments your honour.
                # TODO: Can't single parameter ODE's still be complicated?
                if isinstance(parameters, tuple):
                    param_out[f(parameters[0])] = parameters[1]
                elif isinstance(parameters, (int, float)):
                    param_out[self.param_list[0]] = parameters
                else:
                    raise InputError("Input type should either be a tuple of " +
                                     "(str,numeric) or a single numeric value")
            else:
                raise InputError("Expecting a dict, list or a tuple input " +
                                 "because there are a total of " +
                                 str(self.num_param)+ " parameters")
        else:
            if self.num_param != 0:
                raise Warning("Did not set the values of the parameters. " +
                              "Input was None.")

        self._parameters = param_out

        # unroll the parameter values into the appropriate list
        # if self._paramValue is None or isinstance(self._paramValue, list):
        #     self._paramValue = dict()
        self._paramValue = [0]*len(self._paramList)

        for key, val in self._parameters.items():
            index = self.get_param_index(key)
            self._paramValue[index] = val

        self.set_sp()

    @property
    def state(self):
        """
        Returns
        -------
        list
            state in symbol with current value,
            (:mod:`sympy.core.symbol`,numeric)

        """
        return self._state

    @state.setter
    def state(self, state):
        """
        Set the current value for the states and match it to the
        corresponding symbol

        Parameters
        ----------
        state: array like
            either a vector of numeric value or a
            tuple of two elements, (string, numeric)

        """
        err_str = "Input state is of an unexpected type - " + type(state)

        if state is not None:
            if isinstance(state, (list, tuple)):
                if isinstance(state[0], tuple):
                    self._state = []
                    for s in state:
                        self._state += [(self._extractStateSymbol(s[0]), s[1])]
                else:
                    self._state = self._unrollState(state)
            elif isinstance(state, np.ndarray):
                self._state = self._unrollState(state)
            elif isinstance(state, Number):
                self._state = self._unrollState(state)
            else:
                raise InputError(err_str)
            
            self.set_sp()
        else:
            raise InputError(err_str)

    @property
    def time(self):
        """
        The current time in the ode system

        Returns
        -------
        numeric

        """
        return self._time

    # beware of the fact that
    # time = numeric
    # t = sympy symbol
    @time.setter
    def time(self, time):
        """
        Set the time for the ode system

        Parameters
        ----------
        time: numeric
            Current time of the ode

        """
        if time is not None:
            self._time = time

    @property
    def state_list(self):
        """
        Returns a list of the states in symbol

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return self._stateList

    @state_list.setter
    def state_list(self, state_list):
        """
        Set the set of states for the ode system

        Parameters
        ----------
        stateList: list
            list of string, each string is the name of the state

        """
        if isinstance(state_list, (list, tuple)):
            for s in state_list:
                self._addStateSymbol(s)
        elif isinstance(state_list, (str, ODEVariable)):
            self._addStateSymbol(state_list)
        else:
            raise InputError("Expecting a list")

        self._hasNewTransition.trip()

    @property
    def param_list(self):
        """
        Returns a list of the parameters in symbol

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return self._paramList

    @param_list.setter
    def param_list(self, param_list):
        """
        Set the set of parameters for the ode system

        Parameters
        ----------
        paramList: list
            list of string, each string is the name of the parameter

        """
        if isinstance(param_list, (list, tuple)):
            for p in param_list:
                self._addParamSymbol(p)
        elif isinstance(param_list, (str, ODEVariable)):
            self._addParamSymbol(param_list)
        else:
            raise InputError("Expecting a list")

        self._hasNewTransition.trip()

    @property
    def derived_param_list(self):
        """
        Returns a list of the derived parameters in symbol

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return self._derivedParamList

    @derived_param_list.setter
    def derived_param_list(self, derived_param_list):
        """
        Set the set of derived parameters for the ode system

        Parameters
        ----------
        derived_param: list
            list of string, each string is the name of the derived parameter
            which uses the original parameter
        """
        for param in derived_param_list:
            self._addDerivedParam(param[0], param[1])

    @property
    def transition_list(self):
        """
        Returns a list of the transitions

        Returns
        -------
        list
            with elements as :class:`.Transition`

        """
        return self._transitionList
        # if self._explicitOde is False:
        #     return self._transitionList
        # else:
        #     raise OutputError("ode was defined explicitly, no " +
        #                       "transition available")

    # also need to make it transitionScript class
    @transition_list.setter
    def transition_list(self, transition_list):
        """
        Set the set of transitions for the ode system

        Parameters
        ----------
        transition: list
            list of :class:`.Transition` of type transition in
            :class:`.transition_type`
        """

        # TODO: This warning can be really annoying, I want it to just appear once.
        # print("Update: In the latest version, between state transitions should be passed to SimulateODE"+
        #       " via the Event objects.")

        if isinstance(transition_list, (list, tuple)):
            for t in transition_list:
                self.add_transition(t)
        else:
            raise InputError("Expecting a list")

    @property
    def event_list(self):
        """
        Returns a list of the events

        Returns
        -------
        list
            with elements as :class:`.Transition`

        """
        return self._eventList

    # also need to make it transitionScript class
    @event_list.setter
    def event_list(self, event_list):
        """
        Set the set of events for the ode system

        Parameters
        ----------
        event: list
            list of :class:`.Event`
        """
        if isinstance(event_list, (list, tuple)):
            for event in event_list:
                self.add_event(event)
        else:
            raise InputError("Expecting a list")


    @property
    def birth_death_list(self):
        """
        Returns a list of the birth or death process

        Returns
        -------
        list
            with elements as :class:`.Transition`

        """
        if self._explicitOde is False:
            return self._birthDeathList
        else:
            raise OutputError("ode was defined explicitly, " +
                              "no birth or death process available")

    @birth_death_list.setter
    def birth_death_list(self, birth_death_list):
        """
        Set the set of transitions for the ode system

        Parameters
        ----------
        birth_death: list
            list of :class:`.Transition` of type birth or death in
            :class:`.transition_type`

        """

        # TODO: This warning can be really annoying, I want it to just appear once.
        # print("Update: In the latest version, birth/death transitions should be passed to SimulateODE"+
        #       " via the Event objects.")
        
        if isinstance(birth_death_list, (list, tuple)):
            for bd in birth_death_list:
                self.add_birth_death(bd)
        elif isinstance(birth_death_list, Transition):
            self.add_birth_death(birth_death_list)
        else:
            raise InputError("Input not as expected.  It is not a list " +
                             "or a Transition")

    @property
    def ode_list(self):
        """
        Returns a list of the ode

        Returns
        -------
        list
            with elements as :class:`.Transition`

        """
        return self._odeList
        # if self._explicitOde is True:
        #     return self._odeList
        # else:
        #     raise OutputError("ode was not defined explicitly")

    @ode_list.setter
    def ode_list(self, ode_list):
        """
        Set the set of ode

        Parameters
        ----------
        ode: list
            list of :class:`.Transition` of type birth or death in
            :class:`.transition_type`

        """
        if isinstance(ode_list, list):
            for o in ode_list:
                self.add_ode(o)
        elif isinstance(ode_list, Transition):
            # if it is not a list, then at least it should be an object
            # of the correct type
            self.add_ode(ode_list)
        else:
            raise InputError("Input not as expected.  It is not a list " +
                             "or a Transition")

    @property
    def num_state(self):
        """
        Returns the number of state

        Returns
        -------
        int
            the number of states

        """
        return len(self._stateList)

    @property
    def num_param(self):
        """
        Returns the number of parameters

        Returns
        -------
        int
            the number of parameters

        """
        return len(self._paramList)

    @property
    def num_derived_param(self):
        """
        Returns the number of derived parameters

        Returns
        -------
        int
            the number of derived parameters

        """
        return len(self._derivedParamList)

    # @property
    # def num_pure_transitions(self):
    #     """
    #     Returns the total number of pure transition objects

    #     Returns
    #     -------
    #     int
    #         total number of pure transitions
    #     """
    #     return len(self._transitionList)

    @property
    def num_events(self):
        """
        Returns the total number of pure transition objects

        Returns
        -------
        int
            total number of pure transitions
        """
        return len(self.event_list)

    @property
    def num_birth_deaths(self):
        """
        Returns the total number of birth and death objects

        Returns
        -------
        int
            total number of birth and death processes
        """
        return len(self._birthDeathList)

    @property
    def num_transitions(self):
        """
        Returns the total number of transition objects that belongs to
        either a pure transition or a birth/death process

        Returns
        -------
        int
            total number of transitions
        """

        return self.num_pure_transitions + self.num_birth_deaths

    ###########################################################################


    def _get_model_str(self):
        model_str = "(%s, %s, %s, %s, %s, %s)" % (self._stateList,
                                                  self._paramList,
                                                  self._derivedParamEqn,
                                                  self._transitionList,
                                                  self._birthDeathList,
                                                  self._odeList)
        if hasattr(self, "_parameters"):
            model_str += ".setParameters(%s)" % \
                        {str(k): v for k, v in self._parameters.items()}
        return model_str

    def _add_list_attr(self, attr, attr_list_name):
        """
        Given an attribute (name attr_name), which is a string of comma
        or space separated values, create a new attribute (name attr_name_list)
        which is a list of those separated values.
        e.g. "a,b,c d ef" returns [a, b, c, d, ef]
        """
        if attr is not None:
            if isinstance(attr, str):
                attr = re_split_string.split(attr)
                attr = filter(lambda x: not len(x.strip()) == 0, attr)
            self.__setattr__(attr_list_name, list(attr))
        else:
            raise InputError("No attribute passed to function")

    def _add_list_attr_with_limits(self, attr, attr_list_name):
        """
        Given an attribute (name attr), which is a list 
        , create a new attribute (name attr_list_name)
        which is a list of those separated values.
        e.g. "a,b,c d ef" returns [a, b, c, d, ef]
        """
        if attr is not None:
            if isinstance(attr, str):
                attr = re_split_string.split(attr)
                attr = filter(lambda x: not len(x.strip()) == 0, attr)
                attr_list=list(attr)
                lim_list=[(0, None)]*len(attr_list)
            elif isinstance(attr, list):
                attr_list=[]
                lim_list=[]
                for att in attr:
                    if isinstance(att, tuple):
                        if len(att)!=2:
                            raise InputError("Variable must be tuple of length 2")
                        else:
                            if not isinstance(att[0], str):
                                raise InputError("Variable must be of type string")
                            elif len(att[0].strip()) == 0:
                                raise InputError("Variable has no name")
                            elif not isinstance(att[1], tuple):
                                raise InputError("Limits must be type tuple")
                            elif len(att[1])!=2:
                                raise InputError("Limit tuple must be length 2")
                            else:
                                attr_list.append(att[0])
                                lim_list.append(att[1])
                    elif isinstance(att, str):
                        if len(att.strip()) == 0:
                            raise InputError("Variable has no name")
                        else:
                            attr_list.append(att)
                            lim_list.append( (0, None) )   # We assume that the minimum value of each variable is zero
                    elif isinstance(att, ODEVariable):
                        attr_list.append(att)
                        lim_list.append( (0, None) )
                    else:
                        raise InputError("List elements should be tuple, string or ODEVariable")
            # else:
            #     raise InputError("Input type should either be a string or list")

            self._state_lims=lim_list                           # TODO: maybe assigning limits via a dict is tidier/safer
            self.__setattr__(attr_list_name, list(attr_list))

        else:
            raise InputError("No attribute passed to function")

    def set_sp(self):
        '''
        Set sp attribute, which is collection of all states and vars
        TODO: testing this out still
        '''
        self._s = self._stateList + [self._t]
        self._sp = self._s + self._paramList

        # Calls to the autowrap method can't take ODEVariable class objects
        # Better to convert the objects in self._sp back to sympy objects
        # This code will convert any ODEVariable object in either the stateDict
        # or paramDict dictonary
        for i, item in enumerate(self._sp):
            try:
                 self._sp[i] = self._stateDict[item.ID]
            except Exception:
                 pass
            try:
                 self._sp[i] = self._paramDict[item.ID]
            except Exception:
                 pass
            
        return None

    def get_state_index(self, input_str):
        """
        Finds the index of the state

        Returns
        -------
        int
            the index of the desired state

        """
        if isinstance(input_str, sympy.Symbol):
            return self._extractStateIndex(str(input_str))
        elif isinstance(input_str, ODEVariable):
            return self._extractStateIndex(input_str.ID)
        else:
            return self._extractStateIndex(input_str)

    def get_param_index(self, input_str):
        """
        Finds the index of the parameter

        Returns
        -------
        int
            the index of the desired parameter
        """
        if isinstance(input_str, str):
            return self._extractParamIndex(input_str)
        elif isinstance(input_str, sympy.Symbol):
            return self._extractParamIndex(str(input_str))
        elif isinstance(input_str, ODEVariable):
            return self._extractParamIndex(input_str.ID)
        elif isinstance(input_str, (list, tuple)):
            out_str = [self._extractParamIndex(p) for p in input_str]
            return out_str

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

    def _addSymbol(self, input_str):
        assert re_math.search(input_str) is None, \
            "Mathematical operators not allowed in symbol definition"
        assert re_underscore.search(input_str) is None, \
            "A symbol cannot have underscore as first character"

        if isinstance(input_str, (list, tuple)):
            if len(input_str) == 2:
                if str(input_str[1]).lower() in ("complex", "false"):
                    is_real = 'False'
                elif str(input_str[1]).lower() in ("real", "true"):
                    is_real = 'True'
                else:
                    raise InputError("Unexpected second argument for symbol")
            else:
                raise InputError("Unexpected number of argument for symbol")
        elif isinstance(input_str, str):  # assume real unless stated otherwise
            is_real = 'True'
        else:
            raise InputError("Unexpected input type for symbol")

        assert input_str != 'lambda', "lambda is a reserved keyword"
        tempSym = eval("symbols('%s', real=%s)" % (input_str, is_real))

        if isinstance(tempSym, sympy.Symbol):
            return tempSym
        elif isinstance(tempSym, tuple):
            assert len(tempSym) != 0, "Input symbol is not valid"
            # extract the name of the symbol
            symbolStr = re_symbol_name.search(input_str).group()
            self._vectorStateDict[symbolStr] = tempSym
            return list(tempSym)
        else:
            raise InputError("Unexpected result using the input string:"
                             + str(tempSym))

    def _addStateSymbol(self, input_str):
        if isinstance(input_str, str):
            var_obj = ODEVariable(input_str, input_str)
        elif isinstance(input_str, ODEVariable):
            var_obj = input_str

        symbol_name = self._addSymbol(var_obj.ID)

        if isinstance(symbol_name, sympy.Symbol):
            if str(symbol_name) not in self._paramList:
                self._addVariable(symbol_name, var_obj, self._stateList, self._stateDict)
        else:
            for sym in symbol_name:
                self._addStateSymbol(str(sym))

    def _addParamSymbol(self, input_str):
        # turn input_str into a ODEVarialbe if required
        if isinstance(input_str, str):
            var_obj = ODEVariable(input_str, input_str)
        elif isinstance(input_str, ODEVariable):
            var_obj = input_str

        symbol_name = self._addSymbol(var_obj.ID)

        if isinstance(symbol_name, sympy.Symbol):
            if str(symbol_name) not in self._paramList:
                self._addVariable(symbol_name, var_obj, self._paramList, self._paramDict)
        else:
            for sym in symbol_name:
                self._addParamSymbol(str(sym))

    def _addDerivedParam(self, name, eqn):
        var_obj = ODEVariable(name, name)
        fixed_eqn = checkEquation(eqn, *self._getListOfVariablesDict())
        self._addVariable(fixed_eqn, var_obj, self._derivedParamList, self._derivedParamDict)

        self._hasNewTransition.trip()
        self._derivedParamEqn += [(name, eqn)]
        return None

    def _addVariable(self, symbol, var_obj, obj_list, obj_dict):
        assert isinstance(var_obj, ODEVariable), "Expecting type odeVariable"
        obj_list.append(var_obj)
        obj_dict[var_obj.ID] = symbol

    def add_transition(self, transition):
        """
        Add a single transition between two states

        Parameters
        ----------
        transition: :class:`.Transition`
            The transition object that contains all the information
            regarding the transition
        """

        # Manipulate transitions into events, to allow backwards compatibility.

        if isinstance(transition, Transition):
            if transition.transition_type is TransitionType.T:

                trans=Transition(origin=transition.origin,
                                 destination=transition.destination,
                                 transition_type="T")

                event=Event(rate=transition.equation,
                            transition_list=[trans])

                self._eventList.append(event)
                self._transitionList.append(transition)
                self._hasNewTransition.trip()
            else:
                raise InputError("Input is not a transition between two states")
        else:
            raise InputError("Input %s is not a Transition." % type(transition))

        return None
    
    def add_event(self, event):
        """
        Add an event

        """
        if isinstance(event, Event):
            self._eventList.append(event)
            self._hasNewTransition.trip()
        elif isinstance(event, Transition):             # Convert single transition into event
            rate=event.equation
            event._equation=None
            derived_event=Event(rate=rate,
                                transition_list=[event])
            self._eventList.append(derived_event)
            self._hasNewTransition.trip()
        else:
            raise InputError("Input %s is not an Event or Transition." % type(event))

    def add_birth_death(self, birth_death):
        """
        Add a single birth or death process

        Parameters
        ----------
        transition: :class:`.Transition`
            The transition object that contains all the information
            regarding the process

        """

        # Manipulate transitions into events, to allow backwards compatibility.

        if isinstance(birth_death, Transition):
            t = birth_death.transition_type
            if t is TransitionType.B:
                trans_birth=Transition(destination=birth_death.destination, transition_type="B")

                birth_event=Event(rate=birth_death.equation,
                                  transition_list=[trans_birth])

                self._eventList.append(birth_event)
                self._birthDeathList.append(birth_event)
                self._hasNewTransition.trip()            
            elif t is TransitionType.D:
                trans_death=Transition(origin=birth_death.origin, transition_type="D")

                death_event=Event(rate=birth_death.equation,
                                  transition_list=[trans_death])

                self._eventList.append(death_event)
                self._birthDeathList.append(death_event)
                self._hasNewTransition.trip()   
            else:
                raise InputError("Input is not a birth death process")
        else:
            raise InputError("Input type is not a Transition")

        return None

    def add_ode(self, eqn):
        """
        Add an ode

        Parameters
        ----------
        eqn: :class:`.Transition`
            The transition object that contains all the information
            regarding the ode
        """
        # TODO: check whether previous ode for the same state exist
        # determine if the input object is of the correct type
        if isinstance(eqn, Transition):
            # then whether it is actually an ode
            if eqn.transition_type is TransitionType.ODE:
                # YES!!!
                self._explicitOde = True
                # add to the list
                self._odeList.append(eqn)
            else:
                raise InputError("Input is not a transition of an ode")
        else:
            raise InputError("Input type is not a Transition")

        return None

    def get_TransitionMatrix(self):
        """
        Computes the pure transition matrix given the transitions
        """
        # holders
        self._transitionMatrix = sympy.zeros(self.num_state, self.num_state)

        # Loop through event transitions and only consider pure ones between 2 states
        for event in self.event_list:
            rate=checkEquation(event.rate, *self._getListOfVariablesDict())
            for transition in event.transition_list:
                magnitude=checkEquation(transition._magnitude, *self._getListOfVariablesDict())
                rate_of_change=magnitude*rate
                if transition.transition_type==TransitionType.T:
                    origin_index=self.state_list.index(transition.origin)
                    destination_index=self.state_list.index(transition.destination)
                    self._transitionMatrix[origin_index, destination_index] += rate_of_change

        return self._transitionMatrix

    def get_BirthDeathVector(self):
        # holder
        self._birthDeathVector = sympy.zeros(self.num_state, 1)
        # Extract all info from events
        for event in self.event_list:
            rate=checkEquation(event.rate, *self._getListOfVariablesDict())
            for transition in event.transition_list:
                magnitude=checkEquation(transition._magnitude, *self._getListOfVariablesDict())
                rate_of_change=magnitude*rate
                if transition.transition_type==TransitionType.B:
                    destination_index=self.state_list.index(transition.destination)
                    self._birthDeathVector[destination_index] += rate_of_change
                elif transition.transition_type==TransitionType.D:
                    origin_index=self.state_list.index(transition.origin)
                    self._birthDeathVector[origin_index] -= rate_of_change

        return self._birthDeathVector

    # def _computeOdeVector(self):
    #     # we are only testing it here because we want to be flexible and
    #     # allow the end user to input more state than initially desired
    #     if len(self.ode_list) <= self.num_state:
    #         self._ode = sympy.zeros(self.num_state, 1)
    #         #fromList, _t, eqn, sec = self._unrollTransitionList(self.ode_list)

    #         unrolled_trans_list= self._unrollTransitionList(self.ode_list)
    #         from_list = unrolled_trans_list["from_list"]
    #         eqn_list = unrolled_trans_list["eqn_list"]

    #         for i, eqn in enumerate(eqn_list):
    #             if len(from_list[i]) > 1:
    #                 raise InputError("An explicit ode cannot describe more " +
    #                                  "than a single state")
    #             else:
    #                 self._ode[from_list[i][0]] = eqn
    #     else:
    #         raise InputError("The total number of ode is %s " +
    #                          "where the number of state is %s" %
    #                          len(self.ode_list), self.num_state)

    #     return None

    def get_EventRateVector(self):
        """
        Get all the transitions into a vector, arranged by state to
        state transition then the birth death processes
        """

        self._eventRateVector = sympy.zeros(self.num_events, 1)
        # Extract all info from events
        for i, event in enumerate(self.event_list):
            self._eventRateVector[i]=checkEquation(event.rate, *self._getListOfVariablesDict())

        return self._eventRateVector

    def get_StateChangeMatrix(self):
        """
        The state change matrix, where
        .. math::
            v_{i,j} = change in state i if transition j occurs
            (this could still be in symbolic form at this stage)
        """
        # container for output
        self._vMat = sympy.zeros(self.num_state, self.num_events)

        for event_index, event in enumerate(self.event_list):
            for transition in event.transition_list:
                magnitude=checkEquation(transition._magnitude, *self._getListOfVariablesDict())
                if transition.transition_type==TransitionType.B:
                    destination_index=self.state_list.index(transition.destination)
                    self._vMat[destination_index, event_index] += magnitude
                elif transition.transition_type==TransitionType.D:
                    origin_index=self.state_list.index(transition.origin)
                    self._vMat[origin_index, event_index] -= magnitude
                elif transition.transition_type==TransitionType.T:
                    origin_index=self.state_list.index(transition.origin)
                    destination_index=self.state_list.index(transition.destination)
                    self._vMat[origin_index, event_index] -= magnitude
                    self._vMat[destination_index, event_index] += magnitude
            
        return self._vMat

    def get_pureOdeVector(self):
        '''
        non transition terms
        '''

        pure_ode = sympy.zeros(self.num_state, 1)
        # Now extract any ODE contributions from ODE type transitions
        for ode in self.ode_list:
            origin_index=self.state_list.index(ode.origin)
            pure_ode[origin_index] += checkEquation(ode.equation, *self._getListOfVariablesDict())

        self._pureOdeVector=pure_ode

        return self._pureOdeVector

    def get_ReactantMatrix(self):
        """
        The reactant matrix, where

        .. math::
            \\lambda_{i,j} = \\left\\{ 1, &if state i is involved in transition j, \\\\
                                       0, &otherwise \\right.
        """
        # declare holder
        self._lambdaMat = np.zeros((self.num_state, self.num_events), int)

        for event_index, event in enumerate(self.event_list):
            for transition in event.transition_list:
                if transition.transition_type==TransitionType.B:
                    destination_index=self.state_list.index(transition.destination)
                    self._lambdaMat[destination_index, event_index] = 1
                elif transition.transition_type==TransitionType.D:
                    origin_index=self.state_list.index(transition.origin)
                    self._lambdaMat[origin_index, event_index] = 1
                elif transition.transition_type==TransitionType.T:
                    origin_index=self.state_list.index(transition.origin)
                    destination_index=self.state_list.index(transition.destination)
                    self._lambdaMat[origin_index, event_index] = 1
                    self._lambdaMat[destination_index, event_index] = 1

        return self._lambdaMat

    ########################################################################
    #
    # State change matrix
    #
    ########################################################################

    # TODO: The folloiwng commented out matrices are not used
    #       consider removing if they are just providing clutter

    # def _computeReactantMatrix(self):
    #     """
    #     The reactant matrix, where

    #     .. math::
    #         \\lambda_{i,j} = \\left\\{ 1, &if state i is involved in transition j, \\\\
    #                                    0, &otherwise \\right.
    #     """
    #     # declare holder
    #     self._lambdaMat = np.zeros((self.num_state, self.num_transitions), int)

    #     _f, _t, eqn = self._unrollTransitionList(self._getAllTransition())
    #     for j, eqn in enumerate(eqn):
    #         for i, state in enumerate(self._stateList):
    #             if type(eqn)==int:
    #                 self._lambdaMat[i, j] = 0
    #             elif self._stateDict[state.ID] in eqn.atoms():
    #                 self._lambdaMat[i, j] = 1

    #     return self._lambdaMat

    # # Might replace _computeReactantMatrix. This function gives a matrix 
    # def _computeReactantMatrixOD(self):
    #     """
    #     The alternative reactant matrix, where

    #     .. math::
    #         \\lambda_{i,j} = \\left\\{ 1, &if state i is an origin or destination in transition j, \\\\
    #                                    0, &otherwise \\right.

    #     OD imples this refers to origin and destination
    #     """
        
    #     x=self._vMat!=0
    #     x=x.astype(int)
    #     self._lambdaMatOD=x

    #     return self._lambdaMatOD

    # def _computeDependencyMatrix(self):
    #     """
    #     Obtain the dependency matrix/graph. G_{i,j} indicate whether invoking
    #     the transition j will cause the rate to change for transition j
    #     """
    #     # if self._lambdaMat is None:
    #     #     self._computeReactantMatrix()
    #     # if self._lambdaMatOD is None:
    #     #     self._computeReactantMatrixOD()
    #     if self._vMat is None:
    #         self._computeStateChangeMatrix()

    #     nt = self.num_transitions
    #     self._GMat = np.zeros((nt, nt), int)

    #     for i in range(nt):
    #         for j in range(nt):
    #             d = 0
    #             for k in range(self.num_state):
    #                 d = d or (self._lambdaMat[k, i] and self._vMat[k, j])
    #             self._GMat[i, j] = d

    #     return self._GMat


    ########################################################################
    # Unrolling of the information
    # state
    # TODO: This unrolling is probably not useful anymore if we are
    #       basing the system on events rather than transitions.
    ########################################################################

    def _unrollState(self, state):
        """
        Information unrolling from vector to sympy in state
        """
        state_out = list()
        if self.num_state == 1:
            if isinstance(state, Number):
                state_out.append((self._stateList[0], state))
            else:
                raise InputError("Number of input state not as expected")
        else:
            if len(state) == self.num_state:
                for i, si in enumerate(self._stateList):
                    state_out.append((si, state[i]))
            else:
                raise InputError("Number of input state not as expected")

        return state_out

    def _unrollTransition(self, transition_obj):
        """
        Given a transition object, get the information from it in a usable
        format i.e. indexing within this class
        """
        from_index = self._extractStateIndex(transition_obj.origin)
        to_index = self._extractStateIndex(transition_obj.destination)
        eqn = checkEquation(transition_obj.equation, *self._getListOfVariablesDict())

        
        # Try returning as dict (should improve modularity over tuple output)

        out= {"from_index": from_index,
              "to_index": to_index,
              "eqn": eqn}

        return out

    def _unrollTransitionList(self, transition_list):
        '''
        ...describe...
        '''

        from_list = list()
        to_list = list()
        eqn_list = list()
        type_list = list()

        for transition_obj in transition_list:
            unrolled_transition=self._unrollTransition(transition_obj)
            from_list.append(unrolled_transition["from_index"])
            to_list.append(unrolled_transition["to_index"])
            eqn_list.append(unrolled_transition["eqn"])

        eqn_list = eqn_list if hasattr(eqn_list, '__iter__') else [eqn_list]

        out= {
            "from_list": from_list,
            "to_list": to_list,
            "eqn_list": eqn_list,
            }

        return out

    def _getAllTransition(self, pureTransitions=False):
        '''
        Get all transitions into a list
        If pureTransitions==True just transitions between states
        If pureTransitions==False between states plus birth deaths
        '''
        assert isinstance(pureTransitions, bool), "requires type(pureTransitions) = bool"

        if pureTransitions:
            return self._transitionList
        else:
            return self._transitionList+self._birthDeathList

    def _iterStateList(self):
        """
        Iterator through the states in symbolic form
        """
        for s in self._stateList:
            yield self._stateDict[s.ID]

    def _iterParamList(self):
        """
        Iterator through the parameters in symbolic form
        """
        for p in self._paramList:
            yield self._paramDict[p.ID]

    def _getListOfVariablesDict(self):
        param_dict = [self._paramDict, self._stateDict, self._vectorStateDict]
        return param_dict, self._derivedParamDict

    ########################################################################
    #
    # Ugly shit that is required to fix strings to sympy symbols
    #
    ########################################################################

    def _extractParamIndex(self, input_str):
        if input_str in self._paramDict:
            return self._paramList.index(self._paramDict[input_str])
        else:
            raise InputError("Input parameter: %s does not exist" % input_str)

    def _extractParamSymbol(self, input_str):
        """
        Given a parameter name, input_str
        """
        if isinstance(input_str, ODEVariable):
            input_str = input_str.ID

        if input_str in self._paramDict:
            return self._paramDict[input_str]
        else:
            raise InputError("Input parameter: %s does not exist" % input_str)

    # TODO: figure out why this is so awkward
    def _extractStateIndex(self, input_str):
        '''
        Find the index of the string or sympy.Symbol 'input_str'
        '''
        if input_str is None:
            return list()
        else:
            if isinstance(input_str, (str, sympy.Symbol)):
                input_str = [input_str] # make this an iterable TODO: why?

            if hasattr(input_str, '__iter__'):
                return [self._extractStateIndexSingle(i) for i in input_str]
            else:
                raise Exception("Input must be a string or an iterable " +
                                "object of string")

    def _extractStateIndexSingle(self, input_str):
        '''
        Find the index of the string or sympy.Symbol 'input_str'
        '''
        if isinstance(input_str, ODEVariable):
            return self._stateList.index(input_str)
        else:
            sym_name = self._extractStateSymbol(input_str)
            return self._stateList.index(sym_name)

    def _extractStateSymbol(self, input_str):
        if isinstance(input_str, ODEVariable):
            input_str = input_str.ID

        if input_str in self._stateDict:
            return self._stateDict[input_str]
        else:
            sym_name = re_symbol_name.search(input_str)
            if sym_name is not None:
                if sym_name.group() in self._vectorStateDict:
                    index = re_symbol_index.findall(input_str)
                    if index is not None and len(index) == 1:
                        _i = int(index[0])
                        return self._vectorStateDict[sym_name.group()][_i]
                    else:
                        raise InputError("Cannot find input state, input {} " 
                                         "appears to be a vector that was " 
                                         "not initialized".format(sym_name))
                else:
                    raise InputError("Cannot find input state, input {} " 
                                     "likely to be a vector".format(sym_name))
            else:
                raise InputError("Input state: {} does not exist"
                                 "".format(input_str))

    def _extractUpperTriangle(self, A, nrow=None, ncol=None):
        """
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

        """
        if nrow is None:
            nrow = len(A[:, 0])

        if ncol is None:
            ncol = len(A[0, :])

        B = sympy.zeros(nrow, ncol)
        for i in range(0, nrow):
            for j in range(i, ncol):
                B[i,j] = A[i, j]

        return B