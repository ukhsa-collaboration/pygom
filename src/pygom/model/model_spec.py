import sympy
import numpy as np

from .transition import Event, Transition, TransitionType
from ._model_errors import InputError, OutputError
from ._model_verification import checkEquation
from .ode_variable import ODEVariable, State, Parameter
from . import ode_utils

### Main Classes ###
class HasNewTransition(ode_utils.CompileCanary):
    states = []


# TODO: Is there a point at which we trust that all the parameter orders are agreed on?
# We can proceed to pass state/param information to solving/fitting methods as lists
# safe in the knowledge that we can unpack them on the other side. I guess this belongs
# in the stores and we just need to ensure that caching keeps eveything up to date


# TODO: Think about which module is responsible for the unpacking of certain features.
# For example, if I want a list of parameter values from the store, do I build this
# outside of the store or do I add a @property within the store to present it.


# TODO: Do we forbid users from adding N as a mere parameter?

"""
BaseOdeModel becomes modelspec?

This gethers states, params, events and 

"""

class ModelSpec(object):
    """
    This base object stores the defining objects of a compartmental model
    and has functions to verify the build

    Parameters
    ----------
    state: list
        A list of states (string or State)
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


    NOTE: caching

    Stores own caches about their own contents.
    ModelSpec owns caches that combine information from multiple stores.

    """

    def __init__(
            self,
            state=None,
            param=None,
            derived_param=None,
            event=None,
        ):

        self._invalidate_caches()

        self._parameter_store = ode_utils.ParameterStore()
        self._state_store = ode_utils.StateStore()
        self._derived_parameter_store = ode_utils.DerivedParameterStore()

        # we always need time to be a symbol and it should be denoted as t
        # TODO: should be added to state / param store?
        self._t = sympy.symbols('t', real=True)

        ## Parameters ##
        self._create_parameter_store(param)

        ## States ##
        self._create_state_store(state)

        ## Derived Parameters ##
        # this has to go after adding the parameters
        # because it is suppose to be based on the current
        # base parameters.
        # Making the distinction here because it makes a
        # difference when inferring the parameters of the variables
        self._create_derived_parameter_store(derived_param)

        ## Transitions ##
        self.set_events(event)

    # def __repr__(self):
    #     return f'{self.__class__.__name__ } {self._get_model_str()}'

    def _invalidate_caches(self)->None:
        """
        Tell objects that have cached components to reset their caches as
        the underlying system has changed
        """
        
        self._all_symbols_dict = None

    ###########################################################################
    #
    # States and params
    #
    ###########################################################################

    # ----------------------------------
    # Initialising and populating sotres
    # ----------------------------------

    def _create_state_store(self, state_list:list[str|ODEVariable]) -> None:
        """
        Declare and store the parameter names for the compartmental model

        Parameters
        ----------
        state_list: list
            list of strings or ode variables where each is a state of the 
            system
        """
        # create a new empty store
        # self._state_store = ode_utils.StateStore()
        self.add_states(state_list)

    def add_states(self, state_list:list[str|ODEVariable])->None:
        """
        Append additional states to the ode system

        Parameters
        ----------
        state_list: list
            list of strings or ode variables where each is a state to be 
            added
        """
        if state_list is None:
            state_list = []

        self._state_store.add(state_list, self.all_symbols_dict)
        self._invalidate_caches()

    def _create_parameter_store(self, parameter_list:list[str|ODEVariable]) -> None:
        """
        Declare and store the parameter names for the compartmental model

        Parameters
        ----------
        parameter_list: list
            list of strings or ode variables where each is a parameter of the 
            system
        """
        # create a new empty store
        # self._parameter_store = ode_utils.ParameterStore()
        self.add_parameters(parameter_list)

    def add_parameters(self, parameter_list:list[str|ODEVariable])->None:
        """
        Append additional parameters to the ode system

        Parameters
        ----------
        parameter_list: list
            list of strings or ode variables where each is a parameter to be 
            added
        """
        if parameter_list is None:
            parameter_list = []

        self._parameter_store.add(parameter_list, self.all_symbols_dict)
        self._invalidate_caches()

    def _create_derived_parameter_store(self, derived_parameter_list:list[str|ODEVariable]) -> None:
        """
        Declare and store the derivedparameter names for the compartmental model

        Parameters
        ----------
        derived_parameter_list: list
            list of strings or ode variables where each is a derived parameter of the 
            system
        """
        # create a new empty store
        # self._derived_parameter_store = ode_utils.DerivedParameterStore()
        self.add_derived_parameters(derived_parameter_list)

    def add_derived_parameters(self, derived_parameter_list:list[str|ODEVariable])->None:
        """
        Append additional derived parameters to the ode system

        Parameters
        ----------
        derived_parameter_list: list
            list of strings or ode variables where each is a derived parameter to be 
            added
        """
        if derived_parameter_list is None:
            derived_parameter_list = []

        self._derived_parameter_store.add(derived_parameter_list, self.all_symbols_dict)
        self._invalidate_caches()

    # ----------------------------------
    # Setting numerical values:
    #   - Parameters can have values/distributions
    #   - States can have initial conditions
    # ----------------------------------

    # TODO: check which of these are O(1) and O(N) and maybe add to docstring
    # TODO: state needs intiial condition setting


    def set_parameter_values(self, parameters:dict[str: float])->None:
        """
        Set the values for the parameters already defined.

        Parameters
        ----------
        parameters: dict of {parameter_ID: parameter_value} (prefered)
        """
        self._parameter_store.set_values(parameters)


    # ----------------------------------
    # Accessing store properties
    # ----------------------------------

    ## Numeric values ##

    @property
    def parameter_value_list(self):
        """
        Returns
        -------
        list
            Values in list form
        """
        if self._parameter_store.all_values_set:
            return self._parameter_store.id_value_dict

    @property
    def parameter_id_value_dict(self):
        """
        Returns
        -------
        dict
            Values in dict form {str: Number}
        """
        if self._parameter_store.all_values_set:
            return self._parameter_store.id_value_dict

    @property
    def parameter_symbol_value_dict(self):
        """
        Returns
        -------
        dict
            Values in dict form {symbol: Number}
        """
        if self._parameter_store.all_values_set:
            return self._parameter_store.symbol_value_dict

    ## Derived param expressions ##

    @property
    def derived_parameter_id_expression_dict(self):
        """
        Returns
        -------
        dict
            Derived parameter expressions {str:expression}
        """
        return self._derived_parameter_store.id_expression_dict

    @property
    def derived_parameter_symbol_expression_dict(self):
        """
        Returns
        -------
        dict
            Derived parameter expressions {symbol:expression}
        """
        return self._derived_parameter_store.symbol_expression_dict

    ## State limits ##

    @property
    def state_lower_limits(self):
        """
        State lower numerical limits (ready to be passed to e.g. a solver)

        Returns
        -------
        list[float]
            List of state lower limits
        """
        return self._state_store.lower_limit_list

    @property
    def state_upper_limits(self):
        """
        State upper numerical limits (ready to be passed to e.g. a solver)

        Returns
        -------
        list[float]
            List of state upper limits
        """
        return self._state_store.upper_limit_list

    ## Symbols ##

    @property
    def state_symbol_list(self):
        """
        Returns a list of the states in symbol form

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return self._state_store.symbol_list

    @property
    def param_symbol_list(self):
        """
        Returns a list of the parameters in symbol form

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return self._parameter_store.symbol_list

    @property
    def derived_param_symbol_list(self):
        """
        Returns a list of the derived parameters in symbol form

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return self._derived_parameter_store.symbol_list

    ## Store counts ##

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
        return len(self._derived_parameter_store)

    ## Namespaces ##

    @property
    def state_dict(self)->dict[str: sympy.Symbol]:
        '''
        State dict {str:symbol}
        '''
        return self._state_store.symbol_dict

    @property
    def parameter_dict(self)->dict[str: sympy.Symbol]:
        '''
        Parameter dict {str:symbol}
        '''
        return self._parameter_store.symbol_dict

    @property
    def derived_parameter_dict(self)->dict[str: sympy.Symbol]:
        '''
        Derived parameter dict {str:symbol}
        '''
        return self._derived_parameter_store.symbol_dict

    # Combinations need to pay attention to cache

    @property
    def all_symbols_dict(self)->dict[str: sympy.Symbol]:
        '''
        An attribute collecting together all the states and variables in sympy
        form. This is used for the check equation function.
        '''
        if self._all_symbols_dict is None:
            self._all_symbols_dict = (
                self.state_dict |
                self.parameter_dict |
                self.derived_parameter_dict |
                {'t': self._t}
            )

        return self._all_symbols_dict


    ## Index look up ##

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

    def get_parameter_index(self, input_str:str)->int:
        """
        Finds the index of the parameter

        Returns
        -------
        int
            the index of the desired parameter
        """
        if isinstance(input_str, str):
            return self._parameter_store.get_index(input_str)
        elif isinstance(input_str, (tuple, list)):
            return [self._parameter_store.get_index(x) for x in input_str]

    def get_derived_parameter_index(self, input_str:str)->int:
        """
        Finds the index of the derived parameter

        Returns
        -------
        int
            the index of the desired derived parameter
        """
        if isinstance(input_str, str):
            return self._derived_parameter_store.get_index(input_str)
        elif isinstance(input_str, (tuple, list)):
            return [self._derived_parameter_store.get_index(x) for x in input_str]




    ##########################################################
    #
    # Transitions and Events
    #
    ##########################################################

    # TODO: Transition store depends on some featurs of params,
    # does this hierarchy change anything?

    # TODO: Event indexing might be necessary, can we guarantee everything is in order always?

    ###################################################################################
    ## Store initialiser

    def set_events(self, event_list:list[Transition|Event]|None)->None:
        """
        Declare the transitions for the ode system

        Parameters
        ----------
        state_list: list
            list of strings or ode variables where each is a parameter of the 
            system
        """
        if event_list is None:
            event_list = []

        # create a new store to replace the existing (if creations succeds)
        new_event_store = ode_utils.EventStore()
        new_event_store.add(event_list, self.all_symbols_dict, self.state_dict)
        
        self._event_store = new_event_store
        # self._invalidate_caches()
        # TODO: does not have any bearing on states or params, maybe cache in store itself
        # needs to do something

    @property
    def event_list(self):
        """
        Returns a list of the events

        Returns
        -------
        list
            with elements as :class:`.Transition`

        """
        return self._event_store.event_list

    ###################################################################################
    ## Store counts

    @property
    def num_event(self):
        """
        Returns the number of events

        Returns
        -------
        int
            the number of events

        """
        return self._event_store.num_events

    # TODO: This doesn't actually contain any extra info than the reaction matrix
    #       consider what to do with it.
    # def get_ReactantMatrix(self):
    #     """
    #     The reactant matrix, where

    #     .. math::
    #         \\lambda_{i,j} = \\left\\{ 1, &if state i is involved in transition j, \\\\
    #                                    0, &otherwise \\right.
    #     """
    #     # declare holder
    #     self._lambdaMat = np.zeros((self.num_state, self.num_events), int)

    #     for event_index, event in enumerate(self.event_list):
    #         for transition in event.transition_list:
    #             if transition.transition_type==TransitionType.B:
    #                 destination_index = self.get_state_index(transition.destination)
    #                 self._lambdaMat[destination_index, event_index] = 1
    #             elif transition.transition_type==TransitionType.D:
    #                 origin_index = self.get_state_index(transition.origin)
    #                 self._lambdaMat[origin_index, event_index] = 1
    #             elif transition.transition_type==TransitionType.T:
    #                 origin_index = self.get_state_index(transition.origin)
    #                 destination_index = self.get_state_index(transition.destination)
    #                 self._lambdaMat[origin_index, event_index] = 1
    #                 self._lambdaMat[destination_index, event_index] = 1

    #     return self._lambdaMat
