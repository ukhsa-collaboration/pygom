"""

    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    This module contains the classes required to translate inputs in string
    into an algebraic machine using sympy

"""
# string evaluation
# import re
# from numbers import Number

import sympy
import numpy as np
#from sympy import symbols
#from scipy.stats._distn_infrastructure import rv_frozen

from .transition import Event, Transition, TransitionType
from ._model_errors import InputError, OutputError
from ._model_verification import checkEquation
from .ode_variable import ODEVariable
# from .maths import EventRateVector, PureOdeVector
from . import ode_utils

### Main Classes ###
class HasNewTransition(ode_utils.CompileCanary):
    states = []

class BaseODEModel(object):
    """
    This base object stores the defining objects of a compartmental model
    and has functions to verify the build

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
    _maths_methods = [
        EventRateVector,
        PureOdeVector
    ]

    def __init__(
            self,
            state=None,
            param=None,
            derived_param=None,
            transition=None,
            event=None,
            birth_death=None,
            ode=None,
            # Technical arguments
            backend='lambda'
        ):
        """
        Constructor
        """

        # Setup the maths methods compiler
        # Note that we need the class because we
        # compile both the formatted and unformatted version.
        # Need a manual override of backend because it is possible that we
        # want to perform simulation in a parallel/distributed manner
        # and there are issues with pickling fortran objects
        self._SC = ode_utils.compileCode(backend=backend)

        ## Parameters ##
        self._parameter_store = None
        self.set_parameters(param)

        ## States ##
        self._state_store = None
        self.set_states(state)

        # we always need time to be a symbol and it should be denoted as t
        # TODO: should be added to state / param store?
        self._t = sympy.symbols('t', real=True)

        # self._isDifficult = False

        self._odeList = list()
        self._derivedParamList = list()
        self._derivedParamEqn = list()
        self._derivedParamDict = dict()
        self._vectorStateDict = dict()
        self._transitionList = list()
        self._eventList = list()
        self._birthDeathList = list()
        self._birthDeathVector = list()

        # this has to go after adding the parameters
        # because it is suppose to be based on the current
        # base parameters.
        # Making the distinction here because it makes a
        # difference when inferring the parameters of the variables
        if not ode_utils.none_or_empty_list(derived_param):
            self.derived_param_list = derived_param

        if not ode_utils.none_or_empty_list(event):
            self.event_list = event

        self._spec = ModelSpec(
            states = self._state_store,
            parameters = self._parameter_store,
            derived_parameters = self.derived_param_list,
            events = self.event_list
        )

        if not ode_utils.none_or_empty_list(transition):
            self.transition_list = transition

        if not ode_utils.none_or_empty_list(birth_death):
            self.birth_death_list = birth_death

        if not ode_utils.none_or_empty_list(ode):
            self.ode_list = ode

        self._computeEventRateVector()
        # Add the maths methods to the class

        self._init_maths_methods()
        self._invalidate_caches()

    def __repr__(self):
        return f'{self.__class__.__name__ } {self._get_model_str()}'

    # TODO: model construction
    def _invalidate_caches(self)->None:
        """
        Tell objects that have cached components to reset their caches as
        the underlying system has changed
        """
        # The maths methods
        for mathsmethod in self._maths_methods:
            try:
                method_instance = getattr(self, mathsmethod.method_name)
                method_instance.invalidate_cache()
            except AttributeError:
                pass # We may not yet have all the objects
        
        # The states_and_parameters list (none === not set)
        self._sp = None

    # TODO: model construction    
    def _init_maths_methods(self) -> None:
        """
        Add all the maths method classes as methods to this class
        """
        # Add the maths methods
        for fn_class in self._maths_methods:
            # Create an instance of the maths class with this class as the 
            # associated compartmental model system

            # maths_class_instance = fn_class(parent_model=self)
            maths_class_instance = fn_class(spec = self.model_spec)
            setattr(
                self,
                maths_class_instance.method_name,
                maths_class_instance
            )
            
    ###########################################################################
    #
    # Getters and setters
    #
    ###########################################################################
    # TODO: parameters (check what has changed here)
    @property
    def parameters(self):
        """
        Returns
        -------
        list
            A list which contains tuple of two elements,
            (:mod:`sympy.core.symbol`, numeric)

        """
        if self._parameter_store.all_values_set:
            return [(symb, val) for symb, val in zip(self._parameter_store.symbol_list,
                                                     self._parameter_store.values)]

    # TODO: parameters (check what has changed here)
    @parameters.setter
    def parameters(self, 
                   parameters:dict[str: float]|list[tuple[str,float]]|list[float])->None:
        """
        Set the values for the parameters already defined.  Note that unless
        the parameters are entered via a dictionary or a two element list,tuple
        we assume that it is in the order of :meth:`.getParamList`

        Parameters
        ----------
        parameters: dict of {parameter_ID: parameter_value} (prefered) _or_
            a list which contains elements made of 2 element tuples 
            (string, numeric value) _or_ a single array like object with
            length equal to the number of parameters, in the same order as they
            were created.
        """
        self._parameter_store.values = parameters

    # # TODO: tidy up all this state stuff
    # @property
    # def state(self):
    #     """
    #     Returns
    #     -------
    #     list
    #         state in symbol with current value,
    #         (:mod:`sympy.core.symbol`,numeric)

    #     """
    #     return [(symb, val) for symb, val in zip(self._state_store.symbol_list,
    #                                             self._state_store.values)]

    @property
    def state_list(self):
        """
        Returns a list of the states in symbol

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return self._state_store.symbol_list

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

        self._invalidate_caches()
        #self._hasNewTransition.trip()

    @property
    def param_list(self):
        """
        Returns a list of the parameters in symbol

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return [parameter for parameter in self._parameter_store.index]
    
    # TODO: model/parameter
    def append_parameters(self, parameter_list:list[str|ODEVariable])->None:
        """
        Append additional parameters to the ode system

        Parameters
        ----------
        parameter_list: list
            list of strings or ode variables where each is a parameter to be 
            added
        """
        # create a new store (if we don't already have one)
        if self._parameter_store is None:
            new_parameter_store = ode_utils.ParameterStore()
        else:
            new_parameter_store = self._parameter_store
        new_parameter_store._add_to_store(parameter_list)
        
        self._parameter_store = new_parameter_store
        self._invalidate_caches()

    def set_parameters(self, parameter_list:list[str|ODEVariable]) -> None:
        """
        Declare and store the parameter names for the compartmental model

        Parameters
        ----------
        parameter_list: list
            list of strings or ode variables where each is a parameter of the 
            system
        """
        # TODO: should parameters have limits, like states?

        # create a new store to replace the existing (if creations success)
        new_parameter_store = ode_utils.ParameterStore()
        new_parameter_store.add(parameter_list)
        
        self._parameter_store = new_parameter_store
        self._invalidate_caches()

    def set_states(self, state_list:list[str|ODEVariable])->None:
        """
        Declare the states for the ode system

        Parameters
        ----------
        state_list: list
            list of strings or ode variables where each is a parameter of the 
            system
        """
        # create a new store to replace the existing (if creations succeds)
        new_state_store = ode_utils.StateStore()
        new_state_store.add(state_list)
        
        self._state_store = new_state_store
        self._invalidate_caches()

        # TODO: state limits should be accessed via the state_store upon request
        # limits = np.array(
        #     [self._state_store[v].limits for v in self._state_store.variables],
        #     dtype=float
        # )
        # self.x_min = limits[:, 0]
        # self.x_max = limits[:, 1]

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


    # Transitions and Events

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
        if isinstance(transition_list, (list, tuple)):
            for transition in transition_list:
                self.add_transition(transition)
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

    # @birth_death_list.setter
    # def birth_death_list(self, birth_death_list):
    #     """
    #     Set the set of transitions for the ode system

    #     Parameters
    #     ----------
    #     birth_death: list
    #         list of :class:`.Transition` of type birth or death in
    #         :class:`.transition_type`

    #     """

    #     # TODO: This warning can be really annoying, I want it to just appear once.
    #     # logging.debug("Update: In the latest version, birth/death transitions should be passed to SimulateODE"+
    #     #       " via the Event objects.")
        
    #     if isinstance(birth_death_list, (list, tuple)):
    #         for bd in birth_death_list:
    #             self.add_birth_death(bd)
    #     elif isinstance(birth_death_list, Transition):
    #         self.add_birth_death(birth_death_list)
    #     else:
    #         raise InputError("Input not as expected.  It is not a list " +
    #                          "or a Transition")

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
        return len(self._state_store)

    @property
    def num_param(self):
        """
        Returns the number of parameters

        Returns
        -------
        int
            the number of parameters

        """
        return len(self._parameter_store)

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

    def __str__(self):
        model_str = "(%s, %s, %s, %s, %s, %s)" % (str(self._state_store),
                                                  str(self._parameter_store),
                                                  self._derivedParamEqn,
                                                  self._transitionList,
                                                  self._birthDeathList,
                                                  self._odeList)
        # if hasattr(self, "_parameters"):
        #     model_str += ".setParameters(%s)" % \
        #                 {str(k): v for k, v in self._parameters.items()}
        return model_str

    def _generate_states_and_parameters(self)->None:
        '''
        Creates the states and parameters cache
        '''

        # NOTE: if derived params are included in param list then when compiling it
        #       will try to use expressions (e.g. 'S+I+R') as args in the function definition.
        # self._sp = (self._state_store.symbol_dict | 
        #             {'t': self._t} | 
        #             self._parameter_store.symbol_dict |
        #             self._derivedParamDict)

        self._sp = (
            self._state_store.symbol_dict |
            {'t': self._t} |
            self._parameter_store.symbol_dict
        )

    @property
    def states_and_parameters(self)->list[sympy.Symbol]:
        '''
        An attribute collecting together all the states and variables in sympy
        form. This is used for the autowrap method
        '''

        # TODO: store some namespace of params and symbols and only recalculate if
        #       necessary - i.e. params/states change
        #       

        if self._sp is None:
            self._generate_states_and_parameters()

        return list(self._sp.values())
    
    @property
    def states_and_parameters_dict(self)->dict[str: sympy.Symbol]:
        '''
        An attribute collecting together all the states and variables in sympy
        form. This is used for the check equation function.
        '''
        if self._sp is None:
            self._generate_states_and_parameters()

        return self._sp

    def get_state_index(self, input_str:str)->int:
        """
        Finds the index of the state

        Returns
        -------
        int
            the index of the desired state

        """
        if isinstance(input_str, str):
            return self._state_store.get_index(input_str)
        elif isinstance(input_str, (tuple, list)):
            return [self._state_store.get_index(x) for x in input_str]
        #if isinstance(input_str, sympy.Symbol):
        #     return self._extractStateIndex(str(input_str))
        # elif isinstance(input_str, ODEVariable):
        #     return self._extractStateIndex(input_str.ID)
        # else:
        #     return self._extractStateIndex(input_str)

    def get_param_index(self, input_str:str)->int:
        """
        Finds the index of the parameter

        Returns
        -------
        int
            the index of the desired parameter
        """
        return self._parameter_store.get_index(input_str)

    ########################################################################
    #
    # Setting the scene
    #
    ########################################################################

    def _addDerivedParam(self, name, eqn):
        var_obj = ODEVariable(ID=name)
        fixed_eqn = checkEquation(eqn, self)
        self._addVariable(fixed_eqn, var_obj, self._derivedParamList, self._derivedParamDict)

        self._invalidate_caches()
        #self._hasNewTransition.trip()
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
                event = Event(
                    rate=transition.rate,
                    transition_list=[transition]
                )

                self._eventList.append(event)
                self._transitionList.append(transition)
                self._invalidate_caches()
                #self._hasNewTransition.trip()
            else:
                raise InputError("Input is not a transition between two states")
        else:
            raise InputError("Input %s is not a Transition." % type(transition))

        return None
    
    def add_event(self, event: Event):
        """
        Add an event

        """
        self._eventList.append(event)
        self._invalidate_caches()

    # def add_birth_death(self, birth_death):
    #     """
    #     Add a single birth or death process

    #     Parameters
    #     ----------
    #     transition: :class:`.Transition`
    #         The transition object that contains all the information
    #         regarding the process

    #     """

    #     # Manipulate transitions into events, to allow backwards compatibility.

    #     if isinstance(birth_death, Transition):
    #         t = birth_death.transition_type
    #         if t is TransitionType.B:
    #             trans_birth=Transition(destination=birth_death.destination, transition_type="B")

    #             birth_event=Event(rate=birth_death.equation,
    #                               transition_list=[trans_birth])

    #             self._eventList.append(birth_event)
    #             self._birthDeathList.append(birth_event)
    #             self._invalidate_caches()
    #             #self._hasNewTransition.trip()            
    #         elif t is TransitionType.D:
    #             trans_death=Transition(origin=birth_death.origin, transition_type="D")

    #             death_event=Event(rate=birth_death.equation,
    #                               transition_list=[trans_death])

    #             self._eventList.append(death_event)
    #             self._birthDeathList.append(death_event)
    #             self._invalidate_caches()
    #             #self._hasNewTransition.trip()   
    #         else:
    #             raise InputError("Input is not a birth death process")
    #     else:
    #         raise InputError("Input type is not a Transition")

    #     return None

    # def add_ode(self, eqn):
    #     """
    #     Add an ode

    #     Parameters
    #     ----------
    #     eqn: :class:`.Transition`
    #         The transition object that contains all the information
    #         regarding the ode
    #     """
    #     # TODO: check whether previous ode for the same state exist
    #     # determine if the input object is of the correct type
    #     if isinstance(eqn, Transition):
    #         # then whether it is actually an ode
    #         if eqn.transition_type is TransitionType.ODE:
    #             # YES!!!
    #             self._explicitOde = True
    #             # add to the list
    #             self._odeList.append(eqn)
    #         else:
    #             raise InputError("Input is not a transition of an ode")
    #     else:
    #         raise InputError("Input type is not a Transition")

    #     return None

    def get_TransitionMatrix(self):
        """
        Computes the pure transition matrix given the transitions
        """
        # holders
        self._transitionMatrix = sympy.zeros(self.num_state, self.num_state)

        # Loop through event transitions and only consider pure ones between 2 states
        for event in self.event_list:
            rate=checkEquation(event.rate, self)
            for transition in event.transition_list:
                magnitude=checkEquation(transition._magnitude, self)
                rate_of_change=magnitude*rate
                if transition.transition_type==TransitionType.T:
                    origin_index=self.get_state_index(transition.origin)
                    destination_index=self.get_state_index(transition.destination)
                    self._transitionMatrix[origin_index, destination_index] += rate_of_change

        return self._transitionMatrix

    def get_BirthDeathVector(self):
        # holder
        self._birthDeathVector = sympy.zeros(self.num_state, 1)
        # Extract all info from events
        for event in self.event_list:
            rate=checkEquation(event.rate, self)
            for transition in event.transition_list:
                magnitude=checkEquation(transition._magnitude, self)
                rate_of_change=magnitude*rate
                if transition.transition_type==TransitionType.B:
                    destination_index=self.get_state_index(transition.destination)
                    self._birthDeathVector[destination_index] += rate_of_change
                elif transition.transition_type==TransitionType.D:
                    origin_index=self.get_state_index(transition.origin)
                    self._birthDeathVector[origin_index] -= rate_of_change

        return self._birthDeathVector

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
                    destination_index=self.get_state_index(transition.destination)
                    self._lambdaMat[destination_index, event_index] = 1
                elif transition.transition_type==TransitionType.D:
                    origin_index=self.get_state_index(transition.origin)
                    self._lambdaMat[origin_index, event_index] = 1
                elif transition.transition_type==TransitionType.T:
                    origin_index=self.get_state_index(transition.origin)
                    destination_index=self.get_state_index(transition.destination)
                    self._lambdaMat[origin_index, event_index] = 1
                    self._lambdaMat[destination_index, event_index] = 1

        return self._lambdaMat

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
