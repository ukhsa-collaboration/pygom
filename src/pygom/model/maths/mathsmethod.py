import logging
import numpy as np

from .._model_errors import InputError

# from ..simulate import SimulateOde

from abc import ABC, abstractmethod

# TODO: we have to be very careful about derived parameters.
#       e.g. if N = S + I + R, it is not a constant param but has state dependence.
#       e.g. if L = beta*S*I/N it involves states and params.
#       probably need to ensure that derived params appear in answer too?


# TODO: substituting things in and out is expensive, should cache expanded/compressed expressions 

# class MathsMethod:
class NumericMethod:
    """
    A class designed to be attached to a model object the primary purpose is to 
    produce a numerical evaluation. The symbolic version will be compiled and 
    cached. By default you will need to provide the system state (as time and 
    state values) to perform the evaluation. Parameter numeric values also need
    to have been defined, but these are stored in the parameter store.
    """
    # Will store the compiled function in child classes
    _compiled_obj = None 
    _raw_fn = None

    # The type of output (matrix or vector) that the compiled expression 
    # produces, if None this will be determined automatically
    outType = None 

    # Should be overloaded in child classes to the method name that the class 
    # attaches.
    method_name = None 
    _cache_valid = False
    _pickleable_compile = False

    # any other methods which this one depends on
    depends_on = list()

    # TODO: if we try to make sure that the object is SimulateODE type then we
    #       get a circular import error. I think this indicates design flaw
    #       Goal is to
    #       1) Remove cache ownership from math objects to base pygom
    #       2) Hand over model spec and other math functions via spec member of pygom, no
    #       the entire model

    # def __init__(self, parent_model: SimulateOde)->None:
    def __init__(self, model_spec, compiler, method_register)->None:
        '''
        Initialise the maths method.

        Parameters
        ----------
        model_spec:
            Compartmental model info
        '''
        # Save a pointer to the parent
        self._model_spec = model_spec
        # Use the parent_model's compiler class (don't want each MM having their own).
        self._SC = compiler
        # some methods depend on others, instead of searching the base model, contain
        # math methods in a register
        self._method_register = method_register

        self._derived_params_compiled = None

    def invalidate_cache(self):
        '''
        Marks the cached objects for recreation if called again
        '''
        # TODO: every object needs a copy of the cache?
        self._cache_valid = False

    @abstractmethod
    def __call__(self,):
        '''
        Dunder function so that when added to the model object it acts like a method
        
        Should be overloaded in child classes
        '''
        pass

    @abstractmethod
    def build_expression(self):
        '''
        Build a symbolic form of the maths method

        Returns
        -------
        A sympy object representing the symbolic form of this method
        '''
        pass

    @abstractmethod
    def get_equation(self):
        '''
        Give a symbolic form of the maths method

        Returns
        -------
        A sympy object representing the symbolic form of this method
        '''
        pass

    def expand_derived_params(self, expr):
        """
        Replace derived parameter symbols with their definitions.

        This is necessary to form correct derivatives.
        Otherwise have to implement the chain rule.
        e.g. sympy might take dN/dI = 0, even thoigh N = S + I + R, dN/dI = 1
        """

        derived_dict = self._model_spec._derived_parameter_store.symbol_to_expression_dict

        print(derived_dict)

        return expr.subs(derived_dict)

    def compress_derived_params(self, expr):
        """
        Replace large subexpressions with derived parameter symbols.
        """

        derived_dict = self._model_spec.derived_parameter_expression_dict

        return expr.subs(derived_dict, simultaneous=True)


    def __call__(self, state, time):
        '''
        Dunder function so that when added to the model object it acts 
        like a standard method.
        
        Parameters
        ----------
        state: The values for the system states
        time: The timepoint to evaluate for
        '''
        # Check to see if we need to compile
        if not self._cache_valid or self._compiled_obj is None:
            self.compile_function()

        # perform the numerical calculation
        return self._compiled_obj(*self._get_eval_param(state, time))
    
    def T(self, time, state):
        '''
        Same as :meth:`__call__` (the main method) but with time as first parameter

        This reordering is useful in the calling of integrate and similar 
        functions.
        '''
        return self.__call__(state, time)

    def compile_function(self) -> None:
        '''
        Compile the symbolic form so that rapid numerical evaluation may occur.
        Transforms the output appropriately into numpy
        '''
        logging.debug(f'Compiling sympy object {self.method_name}.')

        inputExpr = self.get_equation()

        # TODO: states and parameters not including time
        # TODO: find a way to deal with derived params

        self._raw_fn = self._SC.compileExpr(
            inputSymb=self._model_spec.states_and_parameters_list,
            inputExpr=inputExpr,
            backend=None       # set at ODE level
        )
        
        # Update the state
        self._pickleable_compile = True if self._SC._backend == 'lambda' else False
        self._cache_valid = True

    # TODO: should have to go into _parameter_store, everything needed should be accessed from mdoel_spec
    # TODO: every time evaluation is done a list of [states, params, time] is concocted. Maybe need to reintroduce cache to stores?

    def _compile_derived_params(self):
        """
        Compile the expressions for the derived parameters
        """

        derived_params = self._model_spec.derived_parameter_expression_dict
        derived_params_compiled = dict()

        for dp_name, dp_symbolic in derived_params.items():
            derived_params_compiled[dp_name] = self._SC.compileExpr(
                inputSymb=self._model_spec.states_and_parameters_list,
                inputExpr=dp_symbolic,
                backend=None
            )

        self._derived_params_compiled = derived_params_compiled

    def _eval_derived_params(self, state, time):
        """
        Get derived param values.

        A dp may be a function of (state, params, other dps, t) so
        we need to evaluate them in order they were defined (hoping
        that the user defined them in order)
        """

        state_time_param_value_dict = self._get_state_time_param_dict(state, time)

        if self._derived_params_compiled is None:
            self._compile_derived_params()

        # container for evaluated params
        derived_params_value_dict = dict.fromkeys(self._derived_params_compiled, None)

        for dp_name, dp_func in self._derived_params_compiled.items():
            combined_values = {state_time_param_value_dict | derived_params_value_dict}
            derived_params_value_dict[dp_name] = dp_func(*combined_values)

        return derived_params_value_dict

    def _get_state_time_param_dict(self, state:list[float], time:float) -> list[float]:
        """

        Get dict of state, time and param values (lacking derived params)
        
        """
        if state is None or time is None:
            raise InputError("Have to input both state and time")
        elif not self._model_spec._parameter_store.all_values_set:
                raise InputError("Have not set the parameters yet")

        # State dict:
        state_value_dict = dict.fromkeys(self._model_spec.state_dict.keys(), state)

        # Time dict:
        time_dict = {'t': time}

        # Param dict:
        param_value_dict = self._model_spec.parameter_value_dict

        return {state_value_dict | time_dict | param_value_dict}

    def _get_eval_param(self, state, time):

        dp = self._eval_derived_params
        state_param_time = self._get_state_time_param_dict(state, time)

        return {dp | state_param_time}
    
    ## Funcitons  to allow pickling and unpickling
    def __getstate__(self):
        '''
        Grab the class's dict and remove the compiled objects if needed
        '''
        state = self.__dict__.copy()
        
        # Remove those compiled methods that have been added
        if not self._pickleable_compile:
            state['_compiled_obj'] = None 
            state['_raw_fn'] = None
            state['_cache_valid'] = False

        return state

# Keep as if we are going to allow pickling Cython we we need this method.    
    # def __setstate__(self, state):
    #     '''
    #     Restore the classes state with reset of compile status
    #     '''
    #     self.__dict__.update(state)

# class SymbolicMethod(MathsMethod):
#     """
#     A class designed to be attached to a model object the primary purpose is to 
#     produce a symbolic representation. The symbolic representation will be 
#     cached.
#     """
#     _symbolic_function = None
#     def __call__(self):
#         '''
#         Returns the symbolic representation of the method
#         '''
#         # Check to see if we need to compile
#         if not self._cache_valid or self._symbolic_function is None:
#             self._symbolic_function = self.get_equation()
            
#             self._cache_valid = True
        
#         return self._symbolic_function
