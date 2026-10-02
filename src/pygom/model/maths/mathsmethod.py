import logging
import numpy as np

from .._model_errors import InputError

# from ..simulate import SimulateOde

from abc import ABC, abstractmethod

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
        return self._compiled_obj(self._getEvalParam(state, time))
    
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

        self._raw_fn, compileType = self._SC.compileExpr(self._model_spec.states_and_parameters_dict,
                                                         inputExpr,
                                                         backend=None, # set at ODE level
                                                         compileType=True) # get additional info      
        
        numRow = inputExpr.rows
        numCol = inputExpr.cols

        outType = self.outType

        # define the different types of compile
        if self.outType is None:
            if numRow == 1 or numCol == 1:
                outType = "vec"
            else:
                outType = "mat"

        if outType.lower() == "vec":
            if compileType == 'np':
                self._compiled_obj = lambda x: self._raw_fn(*x).ravel()
            else:
                self._compiled_obj = lambda x: np.array(self._raw_fn(*x).tolist(),
                                                        float).ravel()
        elif outType.lower() == "mat":
            if compileType == 'np':
                self._compiled_obj = lambda x: self._raw_fn(*x)
            else:
                self._compiled_obj = lambda x: np.array(self._raw_fn(*x).tolist(), float)
        else:
            raise RuntimeError("Specified type of output not recognized")
        
        # Update the state
        self._pickleable_compile = True if self._SC._backend == 'lambda' else False
        self._cache_valid = True

    def _getEvalParam(self, state:list[float], time:float) -> list[float]:
        if state is None or time is None:
            raise InputError("Have to input both state and time")

        elif not self._parent_model._parameter_store.all_values_set:
                raise InputError("Have not set the parameters yet")

        if hasattr(state, '__iter__'):
            # just in case this isn't a list already
            eval_param = list(state) + [time]
        else:
            eval_param = [state] + [time]

        return eval_param + self._parent_model._parameter_store.values
    
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
