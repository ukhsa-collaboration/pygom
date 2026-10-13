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

# TODO: what changes the cache here?
#       1) Changes to events -> recompile and rederive
#       2) changes to params?

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
    _sympy_expression = None

    # The type of output (matrix or vector) that the compiled expression 
    # produces, if None this will be determined automatically
    # TODO: Probably not needed now we let newer sympy version deal with N-dimensional array output
    outType = None 

    # Should be overloaded in child classes to the method name that the class 
    # attaches.
    method_name = None 
    # _cache_valid = False
    _pickleable_compile = False

    # any other methods which this one depends on
    depends_on = list()

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

    ## Allow pickling and unpickling ##

    def __getstate__(self):
        '''
        Grab the class's dict and remove the compiled objects if needed
        '''
        state = self.__dict__.copy()
        
        # Remove those compiled methods that have been added
        if not self._pickleable_compile:
            state['_sympy_expression'] = None
            state['_sympy_expression_dp_expanded'] = None
            state['_compiled_obj'] = None

        return state

    ## Cache management ##

    def invalidate_cache(self):
        '''
        Marks the cached objects for recreation if called again

        Can be called externally
        '''
        self._sympy_expression = None
        self._sympy_expression_dp_expanded = None
        self._compiled_obj = None 

    ## Symbolic evaluation ##

    def _collapse_derived_parameters(self, expr):
        """
        Collapse the derived params in an expression.

        i.e: 'S+I+R' -> 'N'
        """
        expr_to_param = self._model_spec.derived_parameter_expression_symbol_dict

        return expr.subs(expr_to_param, simultaneous=True)

    def _expand_derived_parameters(self, expr):
        """
        Expand the derived params in an expression.

        i.e: 'N' -> 'S+I+R'
        """        
        param_to_expr = self._model_spec.derived_parameter_symbol_expression_dict

        return expr.subs(param_to_expr, simultaneous=True)

    @abstractmethod
    def _build_expression(self):
        '''
        Build and store the symbolic form of the maths method.

        The derived parameters should be collapsed in this version

        This is the defining feature of each method and is
        overloaded by each child class
        '''
        pass

    def _build_expression_dp_expanded(self):
        """
        Build and store the symbolic form of the maths method but...

        The derived parameters are expanded in this version
        """

        self._sympy_expression_dp_expanded = self._expand_derived_parameters(self._sympy_expression)

    def get_equation(self, expand_derived_params=False):
        '''
        Give a symbolic form of the maths method

        We have 2 versions cached,

        Parameters
        ----------
        expand_derived_params:
            If True, replace derived parameter symbols with their expressions

        Returns
        -------
            A sympy object representing the symbolic form of this method
        '''
        # Check to see if we need to (re)build
        if self._sympy_expression is None:
            self._build_expression()
            # Building an expression invalidates the version with dp expressions subbed in
            # NOTE: we could put this at the end of _build_expression() instead
            self._sympy_expression_dp_expanded = None

        if expand_derived_params:
            if self._sympy_expression_dp_expanded is None:
                self._build_expression_dp_expanded()

            return self._sympy_expression_dp_expanded

        return self._sympy_expression

    ## Compilation ##

    def _compile_function(self) -> None:
        '''
        Compile the symbolic form (which has derived parameters removed)
        so that rapid numerical evaluation may occur.
        
        Sympy takes care of transforming N-dimensional output into TODO: numpy array?
        '''
        logging.debug(f'Compiling sympy object {self.method_name}.')

        # We compile the function with derived parameters subbed in
        # NOTE: This is suboptimal for cases where a common subexpression is
        #       calculated many times, but will require us to think about different
        #       classes of derived params to implement. We'd also like derived params
        #       to form part of the output and this is part of the same task.
        inputExpr = self.get_equation(expand_derived_params=True)

        self._compiled_obj = self._SC.compileExpr(
            inputSymb=self._model_spec.states_time_params_symbol_list,
            inputExpr=inputExpr
        )
        
        # Update the state
        self._pickleable_compile = True if self._SC._backend == 'lambda' else False
        # self._cache_valid = True

    ## Numeric evaluation ##

    def __call__(self, state, time):
        '''
        Dunder function so that when added to the model object it acts 
        like a standard method.
        
        Parameters
        ----------
        state: The values for the system states
        time: The timepoint to evaluate for
        '''
        # Check to see if we need to (re)compile
        if self._compiled_obj is None:
            self._compile_function()

        # perform the numerical calculation
        return self._compiled_obj(self._get_state_time_param_list(state, time))
    
    def T(self, time, state):
        '''
        Same as :meth:`__call__` (the main method) but with time as first parameter

        This reordering is useful in the calling of integrate and similar 
        functions.
        '''
        return self.__call__(state, time)

    def _get_state_time_param_list(self, state:list[float], time:float) -> list[float]:
        """

        Get dict of state, time and param values (lacking derived params)
        
        """
        if state is None or time is None:
            raise InputError("Have to input both state and time")
        
        if not self._model_spec.all_parameter_values_set:
            raise InputError("Have not set all of the parameters yet")

        return state + [time] + self._model_spec.parameter_value_list

    # ## Derived parameters ## TODO: how do we deal with them?

    # # TODO: should have to go into _parameter_store, everything needed should be accessed from mdoel_spec
    # # TODO: every time evaluation is done a list of [states, params, time] is concocted. Maybe need to reintroduce cache to stores?

    # def _compile_derived_params(self):
    #     """
    #     Compile the expressions for the derived parameters
    #     """

    #     derived_params = self._model_spec.derived_parameter_expression_dict
    #     derived_params_compiled = dict()

    #     for dp_name, dp_symbolic in derived_params.items():
    #         derived_params_compiled[dp_name] = self._SC.compileExpr(
    #             inputSymb=self._model_spec.states_and_parameters_list,
    #             inputExpr=dp_symbolic,
    #             backend=None
    #         )

    #     self._derived_params_compiled = derived_params_compiled

    # def _eval_derived_params(self, state, time):
    #     """
    #     Get derived param values.

    #     A dp may be a function of (state, params, other dps, t) so
    #     we need to evaluate them in order they were defined (hoping
    #     that the user defined them in order)
    #     """

    #     state_time_param_value_dict = self._get_state_time_param_dict(state, time)

    #     if self._derived_params_compiled is None:
    #         self._compile_derived_params()

    #     # container for evaluated params
    #     derived_params_value_dict = dict.fromkeys(self._derived_params_compiled, None)

    #     for dp_name, dp_func in self._derived_params_compiled.items():
    #         combined_values = {state_time_param_value_dict | derived_params_value_dict}
    #         derived_params_value_dict[dp_name] = dp_func(*combined_values)

    #     return derived_params_value_dict


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
