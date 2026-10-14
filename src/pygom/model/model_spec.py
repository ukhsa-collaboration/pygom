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


# TODO: As long as we remember that the convention is states + time + params, then
#       we cannot go wrong in the compiled maths functions. Not sure if this is
#       currently 100% safe.

# TODO: Write somthing in the docs about the dangers of adding N as a mere parameter
#       and not a derived parameter

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


    NOTE about caching:
        - Stores own caches about their own contents.
        - ModelSpec owns caches that combine information from multiple stores.

    """

    def __init__(
            self,
            state=None,
            param=None,
            derived_param=None,
            event=None,
            dependency_manager=None
        ):

        ## Dependency manager ##
        self._dependency_manager = dependency_manager

        self._invalidate_caches()

        # Initialise some empty stores
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
        self._event_store = ode_utils.EventStore()
        self._create_event_store(event)

    # def __repr__(self):
    #     return f'{self.__class__.__name__ } {self._get_model_str()}'

    def _invalidate_caches(self)->None:
        """
        Objects which depend on multiple stores
        """
        self._states_time_params_id_list = None
        self._states_time_params_dict = None
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
    def all_parameter_values_set(self):
        """
        Returns
        -------
        bool
            True if all params have numerical values assigned
        """

        return self._parameter_store.all_values_set
            

    @property
    def parameter_value_list(self):
        """
        Returns
        -------
        list
            Values in list form
        """
        
        return self._parameter_store.value_list

    @property
    def parameter_id_value_dict(self):
        """
        Returns
        -------
        dict
            Values in ordered dict form {str: Number}
        """
        
        return self._parameter_store.id_value_dict

    @property
    def parameter_symbol_value_dict(self):
        """
        Returns
        -------
        dict
            Values in dict form {symbol: Number}
        """
        
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

    @property
    def derived_parameter_expression_symbol_dict(self):
        """
        Returns
        -------
        dict
            Derived parameter expressions {expression:symbol}
        """
        return self._derived_parameter_store.expression_symbol_dict

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
    def parameter_symbol_list(self):
        """
        Returns a list of the parameters in symbol form

        Returns
        -------
        list
            with elements as :mod:`sympy.core.symbol`

        """
        return self._parameter_store.symbol_list

    @property
    def derived_parameter_symbol_list(self):
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

    ## Combinations of stores (now need to pay attention to model_spec cache) ##

    # NOTE: we must have the order:
    # 1) states
    # 2) time
    # 3) params
    # Might need to consider a safer way to ensure this 

    @property
    def states_time_params_symbol_list(self)->list[str]:
        '''
        An attribute collecting together all variables in symbol
        form, not including derived parameters.       
        '''
        if self._states_time_params_id_list is None:
            self._states_time_params_id_list = self.state_symbol_list + [self._t] + self.parameter_symbol_list

        return self._states_time_params_id_list

    @property
    def states_time_params_symbols_dict(self)->dict[str: sympy.Symbol]:
        '''
        An attribute collecting together all variables in sympy
        form, not including derived parameters.
        '''
        if self._states_time_params_dict is None:
            self._states_time_params_dict = (
                self.state_dict |
                {'t': self._t} |
                self.parameter_dict
            )

        return self._states_time_params_dict

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


    ## Tags ##

    def get_states_by_tag(self, tag_name)->list[str]:
        '''      
        '''

        return self._state_store.get_id_by_tag(tag_name)

    def reject_states_by_tag(self, tag_name)->list[str]:
        '''      
        '''

        return self._state_store.reject_id_by_tag(tag_name)

    def get_parameters_by_tag(self, tag_name)->list[str]:
        '''      
        '''

        return self._parameter_store.get_id_by_tag(tag_name)

    def reject_parameters_by_tag(self, tag_name)->list[str]:
        '''      
        '''

        return self._parameter_store.reject_id_by_tag(tag_name)

    def get_derived_parameters_by_tag(self, tag_name)->list[str]:
        '''      
        '''

        return self._derived_parameter_store.get_id_by_tag(tag_name)

    def reject_derived_parameters_by_tag(self, tag_name)->list[str]:
        '''      
        '''

        return self._derived_parameter_store.reject_id_by_tag(tag_name)

    ##########################################################
    #
    # Transitions and Events
    #
    ##########################################################

    # TODO: We may still need some categorising of events. When the user inspects the
    #       output of e.g. a simulation, they might not know the index of the event
    #       they are interested in, but they may be interested in e.g. "all events
    #       which increase I".

    ## Store initialiser ##

    def add_events(self, event_list:list[Transition|Event]|None)->None:
        """
        Append additional parameters to the ode system

        Parameters
        ----------
        parameter_list: list
            list of strings or ode variables where each is a parameter to be 
            added
        """
        if event_list is None:
            event_list = []

        self._event_store.add(event_list, self.all_symbols_dict, self.state_dict)

        if self._dependency_manager:
            # Notify the central dependency manager that the events have changed. This
            # is to let the maths methods store know that its cache is invalid.
            # The model_spec doesn't know about anything outside of itself, so this
            # is why we must go via the manager.
            self._dependency_manager.notify_change("events")

        # self._invalidate_caches() NOTE: adding events wont invalidate the model_spec cache
        #                                 perhaps in future it will with more complex models                                  


    def _create_event_store(self, event_list:list[Transition|Event]|None)->None:
        """
        Declare the transitions for the ode system

        Parameters
        ----------
        state_list: list
            list of strings or ode variables where each is a parameter of the 
            system
        """

        self.add_events(event_list)

    # ------------------
    # Properties
    # ------------------

    ## Events ##

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

    ## Counts ##

    @property
    def num_event(self):
        """
        Returns the number of events

        Returns
        -------
        int
            the number of events

        """
        # NOTE: we don't use a len here because maybe we want num_transitions
        #       in which case it's best to be explicit (though these are most
        #       often the same)

        return self._event_store.num_events