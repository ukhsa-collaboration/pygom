"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module/class that contains a variable object for the ode

"""
from sympy.physics.units.quantities import Quantity
from sympy import symbols, Symbol
import numpy as np
import re
import keyword
from ._model_errors import InputError
from numbers import Number
from scipy.stats._distn_infrastructure import rv_frozen
from typing import Callable
from ._model_verification import checkEquation


class CallableParameter:
    def __init__(self, value: tuple[Callable|str, dict|tuple]):
        """
        Data type for parameters which draws a random number when called

        Parameters
        ----------
        value: tuple[callable, dict|tuple]
            value[0] is the probability distribution and value[1] the function parameters
        """

        dist, params = value
        self.source = dist

        # -----------------------------
        # Parse arguments
        # -----------------------------
        if isinstance(params, dict):
            self.args = ()
            self.kwargs = params
        elif isinstance(params, tuple):
            self.args = params
            self.kwargs = {}
        else:
            raise InputError(
                'Second element should be either a tuple or a '
                'dict when using multi-argument distribution '
                f'definition. Type of input was {type(params)}.'
            )

    def __call__(self, rng):
        if isinstance(self.source, str):
            method = getattr(rng, self.source)

            return method(
                *self.args,
                **self.kwargs
            )

        return self.source(
            rng,
            *self.args,
            **self.kwargs
        )

class ODEVariable(object):
    """
    A class that defines the variable meta-data


    TODO: In some parts of the code, it's assumed that ID = str(symbol) - decide if always true
          We cannot use symbol as ID because it sometimes has hidden assumtpions which make lookup nontrivial
          e.g. Symbol('X') != Symbol('X', real=True)

    Parameters
    ----------
    ID: str
        identifier of the variable
    symbol: sympy.Symbol
        sympy symbolic representation of the variable. Often taken to
        be sympy.Symbol(ID), but not necessarily.
    units: str, optional
        what unit the variable takes. Defaults to None.
    real: bool, optional
        if the variable can only be a real number, defaults to True
    limits: tuple(float, float), optional
        minimum and maximum allowed numerical values of the variable
    """
    def __init__(
            self,
            ID:None|str, 
            # symbol:None|Symbol|str=None,
            description:None|str=None,
            units:None|Quantity=None,
            real:bool=True,
            limits:None|tuple=(0, np.inf),
            # value=None,
            tags=None|list[str],
            class_name='ODEVariable'
        ):

        if ID is None:
            raise InputError(
                f"Must specify ID"
            )

        # if (ID is None) and (symbol is None):
        #     raise InputError(
        #         f"Must specify at least one of ID or symbol"
        #     )

        # if ID is None:
        #     ID = str(symbol)

        # if symbol is None:
        #     symbol = ID

        self.ID = ID
        self.real = real
        self.units = units
        self.limits = limits
        # self.value = value
        self.symbol = ID        # Build symbol from string ID
        self.description = description

        self.tags=tags
        self.class_name = class_name

    ######################################################################
    # Dunder methods
    ######################################################################

    def __str__(self)->str:
        return self.ID

    def __repr__(self)->str:
        return (
            f"{self.class_name}("
            f"{self.ID!r}, "
            f"{self.symbol!r}, "
            f"{self.units!r}, "
            f"{self.limits!r})"
        )
                                                
    def __eq__(self, other):
        if isinstance(other, str):
            return self.ID == other
        elif isinstance(other, Symbol):
            return self.symbol == other         # TODO: Symbol('x') != Symbol('x', real=True), do we want to use this as a valid check?
        elif isinstance(other, ODEVariable):
            return (
                self.ID == other.ID and \
                self.symbol == other.symbol
            )
        else:
            # raise NotImplementedError('Wrong input type of %s' % type(other))
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __le__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __gt__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    def __ge__(self, other):
        raise NotImplementedError("Only equality comparison allowed")

    ######################################################################
    # Properties, setters, checks
    ######################################################################

    ## id ##

    @property
    def ID(self):
        return self._ID
    
    @ID.setter
    def ID(self, ID:str):
        """
        Set ID to represent variable

        Parameters
        ----------
        ID : str
        """
        if not isinstance(ID, str):
            raise TypeError("ID must be a string")    
        self._ID = ID

    ## symbol ##

    def _generate_symbol(
            self,
            symbol_name: str,
            real:bool=True
        ) -> list:
        """
        Wrapper of sympy.symbols() to generate one or more Sympy symbols from variable name(s)

        We cannot let sympy build symbols on its own since we have some additional requirements.

        Parameters
        ----------
        symbol_name: str
            Name of the symbol(s)
        real: bool
            True if real valued

        Returns
        -------
        sympy.Symbol
        """

        _SYMBOL_RULES = (
            "Symbol names must:\n"
            "  - start with a letter\n"
            "  - then contain only letters, digits or underscores"
            # "  - or be in SymPy range notation (e.g. 'y1:4')"
        )

        _VALID_SYMBOL = re.compile(
            r"^[A-Za-z][A-Za-z0-9_]*$"
        )

        # NOTE: should we make 't' a protected symbol for time?
        #       maybe add t at model initialisation and then any
        #       later attempts to add t will be blocked?
        if keyword.iskeyword(symbol_name):
            raise InputError(
                f"'{symbol_name}' is a reserved Python keyword"
            )
        if not _VALID_SYMBOL.fullmatch(symbol_name):
            raise InputError(
                f"Invalid symbol name '{symbol_name}'.\n{_SYMBOL_RULES}"
            )

        return symbols(symbol_name, real=real)

    @property
    def symbol(self):
        return self._symbol
    
    @symbol.setter
    # def symbol(self, symbol:Symbol|str):
    def symbol(self, symbol:str):
        """
        Set symbol to represent variable

        Parameters
        ----------
        symbol : Symbol|str
        """
        if isinstance(symbol, str):
            symbol = self._generate_symbol(symbol, self.real)
        else:
            raise InputError(
                'The symbol attribute must be of str type'
            )
        # elif not isinstance(symbol, Symbol):
        #     raise InputError(
        #         'The symbol attribute must be of sympy.Symbol or str type'
        #     )
        self._symbol = symbol

    ## limits ##

    @property
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, limits):
        """
        Set upper and lower numerical limits for variable

        Parameters
        ----------
        limits : tuple|list
            Length 2, where limits = (lower, upper)
        """
        if not isinstance(limits, (tuple, list)):
            raise InputError("Limits must be a tuple or list")

        if len(limits) != 2:
            raise InputError(f"Limits should contain exactly 2 values, received {len(limits)}")

        lower, upper = limits

        if (not isinstance(lower, Number)) or (not isinstance(upper, Number)):
            raise InputError("Limits must be numeric")

        if lower >= upper:
            raise InputError("Lower limit must be strictly less than upper limit")

        self._limits = (lower, upper)

    ## Tags ##

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, tags):
        """
        Set state tags. A state may have multiple tags which provide extra
        context on the type of individuals which populate it.
        """
        if tags is None:
            tags = []

        for tag in tags:
            if tag not in self._allowed_tags:
                raise(InputError(
                    f"Invalid state tag: '{tag}'. "
                    f"Choose from: {self._initial_value}"
                    ))

        self._tags = set(tags)

    ## numeric value ##

    def _validate_value(self, value):
        """
        Validate if a proposed numerical value:
        - Is real if required
        - Falls within allowed limits
        """

        # No action required if value being left without a value
        # or if the source (callable) is being declared but not a value. 
        if value is None:
            return
        # if isinstance(value, CallableParameter):
        #     return

        if not isinstance(value, Number):
            raise ValueError(
                f"Numeric value of '{self.ID}' must be type 'Number'."
            )

        if self.real and not np.isreal(value):
            raise ValueError(
                f"Numeric value of '{self.ID}' must be real."
            )

        lower, upper = self.limits

        if value < lower:
            raise ValueError(
                f"Numeric value of '{self.ID}' must be >= {lower}."
            )

        if value > upper:
            raise ValueError(
                f"Numeric value of '{self.ID}' must be <= {upper}."
            )

###########################
# Child classes
###########################

class State(ODEVariable):
    """
    A State is a variable for which values belong to the solver.

    These are different in that they:
    - default lower limit is 0
    - initial_values attribute
    - tags
    # TODO: Deal with tags the same way we handle TransitionType
    """
    def __init__(
            self,
            ID:None|str, 
            # symbol:None|Symbol|str=None,
            units:None|Quantity=None,
            real:bool=True,
            limits:None|tuple=(0, np.inf),
            initial_value:None|Number=None,
            current_value:None|Number=None,
            tags:None|list[str]=None
        ):
        super().__init__(
            ID=ID,
            # symbol=symbol,
            units=units,
            real=real,
            limits=limits,
            tags=tags,
            # value=value,
            class_name='State',
        )

        # TODO: tags should probably be imported from the epi/econ/ecol/whetever module.
        #       these are clearly econ tags for now.

        self._allowed_tags = [
            "alive",
            "dead",
            "infected",
            "infectious",
            "cumulative"
        ]

        self.current_value = current_value
        self.initial_value = initial_value

    ######################################################################
    # Properties and setters
    ######################################################################

    ## Initial values ##

    @property
    def initial_value(self):
        return self._initial_value

    @initial_value.setter
    def initial_value(self, value):
        """
        Set initial value.
        """
        self._validate_value(value)
        self._initial_value = value

    ## Current values ##

    @property
    def current_value(self):
        return self._current_value

    @current_value.setter
    def current_value(self, value):
        """
        Set current value.
        """
        self._validate_value(value)
        self._current_value = value


class Parameter(ODEVariable):
    """
    Parameters:
    - May have callable value types
    - Default limits are -inf, +inf
    """
    def __init__(
            self,
            ID:None|str, 
            # symbol:None|Symbol|str=None,
            units:None|Quantity=None,
            real:bool=True,
            limits:None|tuple=(-np.inf, np.inf),
            value:None|Number|rv_frozen|CallableParameter=None,
            tags:None|list[str]=None
        ):

        super().__init__(
            ID=ID,
            # symbol=symbol,
            units=units,
            real=real,
            limits=limits,
            # value=value,
            tags=tags,
            class_name='Parameter'
        )

        self.source = value
        self.value = value

    ## Source ##

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        """
        Set source of variable:

        Parameters
        ----------
        value : rv_frozen|CallableParameter|Number
            Length 2, where limits = (lower, upper)
        """

        if isinstance(source, (rv_frozen, CallableParameter)):
            self._is_stochastic = True
        else:
            self._is_stochastic = False

        self._source = source   

    @property
    def is_stochastic(self):
        return self._is_stochastic

    ## Value ##

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        """
        Set numeric value of variable, or declare a callable which
        will generate values

        Parameters
        ----------
        value : rv_frozen|CallableParameter|Number
            Length 2, where limits = (lower, upper)
        """
        if isinstance(value, (rv_frozen, CallableParameter)):
            self.source = value
            self._value = None
        else:
            self._validate_value(value)
            self._value = value

    def realise(self, rng=None):
        """
        Generate a realisation of the parameter.
        If static, just return the value.
        If random, generate a new random realisation.
        """

        if self.is_stochastic:
            new_value = self.source(rng)
            self.value = new_value

        return self.value


class DerivedParameter(ODEVariable):
    """
    Derived Parameters:
    - Like states, values are calculated elsewhere and not stored here
    - In fact derived parameters should not let you set their values
    - Has the string_expression, which gives the algebraic definition (in string form
      symbolic form comes after we perform checks elsewhere)
    - Has no value
    """
    def __init__(
            self,
            ID:None|str,
            string_expression:None|str,
            # symbol:None|Symbol|str=None,
            units:None|Quantity=None,
            real:bool=True,
            limits:None|tuple=(-np.inf, np.inf),
            tags:None|list[str]=None
            # value=None
        ):
        super().__init__(
            ID=ID,
            # symbol=symbol,
            units=units,
            real=real,
            limits=limits,
            # value=value,
            tags=tags,
            class_name='DerivedParameter'
        )

        # TODO: work in progress, but may be useful to tag derived parameters in this way:

        self._allowed_tags = [
            'cumulative',
            'dynamic',
            'convenience',
            'subexpression',
            'of_interest'
        ]

        self.string_expression = string_expression


     ## string expression ##

    @property
    def string_expression(self):
        return self._string_expression

    @string_expression.setter
    def string_expression(self, expr):
        """
        This needs to be checked and sympy-ed later on
        We do so when adding to the store
        Store owns each namespace, ModelSpec owns them all
        """
        if isinstance(expr, str):
            self._string_expression = expr
        else:
            raise(
                InputError(
                    "Derived parameter expression must be type 'str',"
                    f"instead received '{type(expr)}'"
                )
            )

    ## sympy expression ##

    @property
    def sympy_expression(self):
        return self._sympy_expression

    def build_sympy_expression(self, all_symbols):
        """
        Build symbolic expression
        """

        self._sympy_expression = checkEquation(
            self.string_expression,
            all_symbols
        )
