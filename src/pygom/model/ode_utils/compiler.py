import logging
import sympy
from sympy.utilities.autowrap import autowrap
from sympy.utilities.lambdify import lambdify
import numpy as np

class CompileCode:
    '''
    A class that compiles an algebraic expression in sympy to a faster
    numerical file using the appropriate backend.
    '''

    # Cache backend as a class attribute to avoid finding
    # backend if we have additional instances
    _cached_backend = None

    def __init__(self, backend=None):
        '''
        Initializing the class. Automatically checks which backend is
        available. Currently only those linked to np are used where
        those linked with Theano are not.
        '''

        # Valid backends (in order of most to least preferable)
        self._valid_backends = [
            "f2py",
            "Cython"
        ]

        if backend is not None:
            if backend not in self._valid_backends + ['lambda']:
                raise ValueError(f"Unknown backend, '{backend}'")
            self._backend = backend
        else:
            if self._cached_backend is None:
                CompileCode._cached_backend = self._find_backend()

            self._backend = CompileCode._cached_backend

    def _find_backend(self):
        logging.debug('Finding available backend.')

        x = sympy.Symbol("x")
        expr = sympy.sin(x) / x

        for backend in self._valid_backends:
            try:
                fn = autowrap(expr, args=[x], backend=backend)
                fn(1)
                return backend

            except Exception:
                continue

        return "lambda"

    def compileExpr(
        self,
        inputSymb,
        inputExpr,
        backend=None
        ):
        '''
        Compiles the expression and determines the backend if required.

        Parameters
        ----------
        inputSymb: list
            the set of symbols for the input expression
        inputExpr: expr
            expression in sympy
        backend: optional
            the backend we want to use to compile

        Returns
        -------
        Compiled function taking arguments of the input symbols
        '''

        if backend is None:
            backend = self._backend

        try:
            if backend in ("f2py", "Cython"):
                raw_fn = autowrap(
                    expr=inputExpr,
                    args=inputSymb,
                    backend=backend
                )
                compile_type = "np"

            elif backend == "lambda":
                raw_fn = lambdify(
                    inputSymb,
                    inputExpr,
                    modules="numpy"
                )
                compile_type = "np"

            else:
                raise ValueError(f"Unknown backend, '{backend}'")

        except Exception:
            try:
                raw_fn = lambdify(
                    inputSymb,
                    inputExpr,
                    modules="mpmath"
                )
                compile_type = "mpmath"

            except Exception:
                raw_fn = lambdify(
                    inputSymb,
                    inputExpr,
                    modules="sympy"
                )
                compile_type = "sympy"

        logging.debug('Compiled expression as {}'.format(compile_type))

        return lambda x: np.asarray(
            raw_fn(*x),
            dtype=float
        )