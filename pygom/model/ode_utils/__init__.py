'''
Utilities used throughout the package.
'''

import math
import logging
from numbers import Number

import numpy as np
import scipy.integrate
import scipy.sparse
import sympy
from sympy.utilities.lambdify import lambdify
from sympy.utilities.autowrap import autowrap

from pygom.model._model_errors import (ArrayError,
                                       ExpressionErrror,
                                       InputError,
                                       IntegrationError
                                       )

from .compile_canary import CompileCanary
from .plot_det import plot_det
from .plot_stoc import plot_stoc
from .checks_and_conversions import (check_array_type,
                                     check_dimension,
                                     is_list_like,
                                     str_or_list,
                                     none_or_empty_list
                                     )

__all__ = [
    'shapeAdjust',
    'integrate',
    'integrateFuncJac',
    'compileCode',
    'CompileCanary',
    #plots
    'plot_det',
    'plot_stoc',
    #checks and conversions
    'check_array_type',
    'check_dimension',
    'is_list_like',
    'str_or_list',
    'none_or_empty_list'
    ]

atol = 1e-10
rtol = 1e-10

class shapeAdjust(object):
    '''
    A class that change vector into matrix and vice versa for
    vectors used in :class:`DeterministicOde`

    Parameters
    ----------
    numState: int
        number of states
    numParam: int
        number of parameters
    numTarget: int, optional
        number of targeted states, default assumes that this is the
        same as numState
    '''

    def __init__(self, numState, numParam, numTarget=None):
        self._d = numState
        self._p = numParam

        if numTarget is None:
            self._m = numState
        else:
            self._m = numTarget

    def vecToMatSens(self, s):
        '''
        Transforms the sensitivity vector to a matrix
        '''
        return vecToMatSens(s, self._d, self._p)

    def vecToMatFF(self, ff):
        '''
        Transforms the forward forward sensitivity vector to a matrix
        '''
        return vecToMatFF(ff, self._d, self._p)

    def matToVecSens(self, S):
        '''
        Transforms the sensitivity matrix to a vector
        '''
        return matToVecSens(S, self._d, self._p)

    def matToVecFF(self, FF):
        '''
        Transforms the forward forward sensitivity matrix to a vector
        '''
        return matToVecFF(FF, self._d, self._p)

    def kronState(self, A, pre=False):
        '''
        A sparse multiplication with an identity matrix of size
        equal to the number of state as initialized

        Parameters
        ----------
        A: array like
            a 2d array
        pre: bool, optional
            If True, then returns :math:`I \\otimes A`.
            If False then :math:`A \\otimes I`, where :math:`A` is the input
            matrix, :math:`I` is the identity matrix and :math:`\\otimes` is
            the kron operator
        '''
        if pre:
            return scipy.sparse.kron(scipy.sparse.eye(self._d), A)
        else:
            return scipy.sparse.kron(A, scipy.sparse.eye(self._d))

    def kronParam(self, A, pre=False):
        '''
        A sparse multiplication with an identity matrix of size
        equal to the number of parameters as initialized

        Parameters
        ----------
        A: array like
            a 2d array
        pre: bool, optional
            If True, then returns :math:`I \\otimes A`.
            If False then :math:`A \\otimes I`, where :math:`A` is the input
            matrix, :math:`I` is the identity matrix and :math:`\\otimes` is
            the kron operator
        '''
        if pre:
            return scipy.sparse.kron(scipy.sparse.eye(self._p), A)
        else:
            return scipy.sparse.kron(A, scipy.sparse.eye(self._p))

def integrate(ode, x0, t, full_output=False):
    '''
    A wrapper on top of :mod:`odeint <scipy.integrate.odeint>` using
    :class:`DeterministicOde <pygom.model.DeterministicOde>`.

    Parameters
    ----------
    ode: object
        of type :class:`DeterministicOde <pygom.model.DeterministicOde>`
    t: array like
        the time points including initial time
    full_output: bool, optional
        If the additional information from the integration is required

    '''

    # INTEGRATE!!! (shout it out loud, in Dalek voice)
    # determine the number of output we want
    solution, output = scipy.integrate.odeint(ode.ode,
                                              x0, t,
                                              Dfun=ode.jacobian,
                                              mu=None, ml=None,
                                              col_deriv=False,
                                              mxstep=10000,
                                              full_output=True)

    if full_output == True:
        # have both
        return solution, output
    else:
        return solution

def integrateFuncJac(func, jac, x0, t0, t, args=(), includeOrigin=False,
                     full_output=False, method=None, nsteps=10000):
    '''
    A replacement for :mod:`scipy.integrate.odeint` which performs integration
    using :class:`scipy.integrate.ode`, tries to pick the correct integration
    method at the start through eigenvalue analysis

    Parameters
    ----------
    func: callable
        the ode :math:`f(x)`
    jac: callable
        jacobian of the ode, :math:`J_{i,j} = \\nabla_{x_{j}} f_{i}(x)`
    x0: `numpy.ndarray` or list of numeric
        initial value of the states
    t0: float
        initial time
    args: tuple, optional
        additional arguments to be passed on
    includeOrigin: bool, optional
        if the output should include the initial states x0
    full_output: bool, optional
        if additional output is required
    method: str, optional
        the integration method.  All those availble in
        :class:`ode <scipy.integrate.ode>` are allowed with 'vode' and
        'ivode' representing the non-stiff and stiff version respectively.
        Defaults to None, which tries to choose the integration method
        via eigenvalue analysis (only one) using the initial conditions
    nstep: int, optional
        number of steps allowed between each time point of the integration

    Returns
    -------
    solution: array like
        a :class:`np.ndarray` of shape (len(t), len(x0)) if includeOrigin is
        False, else an extra row with x0 being the first.
    output : dict, only returned if full_output=True
        Dictionary containing additional output information

        =========  ===========================================
        key        meaning
        =========  ===========================================
        'ev'       vector of eigenvalues at each t
        'maxev'    maximum eigenvalue at each t
        'minev'    minimum eigenvalue at each t
        'suc'      list whether integration is successful
        'in'       name of integrator
        =========  ===========================================

    '''
    # determine the type of integrator we want
    # print "we are in"
    if method is None:
        if full_output==True:
            # obtain the eigenvalue
            e = np.linalg.eig(jac(t0, x0, *args))[0]
            method = _determineIntegratorGivenEigenValue(e)
        else:
            method = 'lsoda'

    r = _setupIntegrator(func, jac, x0, t0, args, method, nsteps)
    # print method
    # print r
    # holder for the integration
    solution = list()
    if full_output:
        successInfo = list()
        eigenInfo = list()
        maxEigen = list()
        minEigen = list()

    if includeOrigin:
        solution.append(x0)

    if isinstance(t, Number):
        # force it to something iterable
        t = [t]
    elif is_list_like(t): #, (np.ndarray, list, tuple)):
        pass
    else:  #
        raise InputError("Type of input time is not of a recognized type")

    for deltaT in t:
        if full_output:
            o1, o2, o3, o4, o5 = _integrateOneStep(r, deltaT,
                                                   func, jac, args, True)
            successInfo.append(o2)
            eigenInfo.append(o3)
            maxEigen.append(o4)
            minEigen.append(o5)
            method = _determineIntegratorGivenEigenValue(o3)
            r = _setupIntegrator(func, jac, o1, deltaT, args, method, nsteps)
        else:
            # TODO: switches
            o1 = _integrateOneStep(r, deltaT, func, jac, args, False)
        # append solution, same thing whether the output is full or not
        solution.append(o1)
    # finish integration

    solution = np.array(solution)

    if full_output == True:
        # have both
        output = dict()
        output['ev'] = np.array(eigenInfo)
        output['minev'] = np.array(minEigen)
        output['maxev'] = np.array(maxEigen)
        output['suc'] = np.array(successInfo)
        output['in'] = method
        return solution, output
    else:
        # only the integration
        return solution

def _integrateOneStep(r, t, func, jac, args=(), full_output=False):
    '''
    Perform integration with just one step
    '''
    r.integrate(t)
    # wish to do checking at each iteration?  A feature to be
    # considered in the future
    if r.successful():
        if full_output:
            e = np.linalg.eig(jac(r.t, r.y, *args))[0]
            return r.y, r.successful(), e, max(e), min(e)
        else:
            return r.y
    else:
        try:
            np.linalg.eig(jac(r.t, r.y, *args))
        except:
            raise IntegrationError("Failed integration with x =  " + str(r.y))
        else:
            a = np.linalg.eig(jac(r.t, r.y, *args))[0]
            raise IntegrationError("Failed integration with x =  " + str(r.y) +
                                   " and max/min eigenvalues of the jacobian is "
                                   + str(max(a)) + " and " + str(min(a)))

def _setupIntegrator(func, jac, x0, t0, args=(), method=None, nsteps=10000):
    if method == 'dopri5':
        # if we are going to use rk5, then one thing for sure is that we
        # know for sure that the set of equations are not stiff.
        # Furthermore, the jacobian information will never be used as
        # it evaluate f(x) directly
        r = scipy.integrate.ode(func).set_integrator('dopri5', nsteps=nsteps,
                                                     atol=atol, rtol=rtol)
    elif method == 'dop853':
        r = scipy.integrate.ode(func).set_integrator('dop853', nsteps=nsteps,
                                                     atol=atol, rtol=rtol)
    elif method == 'vode':
        r = scipy.integrate.ode(func, jac).set_integrator('vode',
                                                          with_jacobian=True,
                                                          lband=None, uband=None,
                                                          nsteps=nsteps,
                                                          atol=atol, rtol=rtol)
    elif method == 'ivode':
        r = scipy.integrate.ode(func, jac).set_integrator('vode', method='bdf',
                                                          with_jacobian=True,
                                                          lband=None, uband=None,
                                                          nsteps=nsteps,
                                                          atol=atol, rtol=rtol)
    elif method == 'lsoda':
        r = scipy.integrate.ode(func, jac).set_integrator('lsoda',
                                                          with_jacobian=True,
                                                          lband=None, uband=None,
                                                          nsteps=nsteps,
                                                          atol=atol, rtol=rtol)
    else:
        r = scipy.integrate.ode(func, jac).set_integrator('lsoda',
                                                          with_jacobian=True,
                                                          lband=None, uband=None,
                                                          nsteps=nsteps,
                                                          atol=atol, rtol=rtol)

    r.set_f_params(*args).set_jac_params(*args)
    r.set_initial_value(x0, t0)
    return r

def _determineIntegratorGivenEigenValue(e):
    # the min and max of them
    maxE = max(e)
    minE = min(e)
    if maxE >= 0:
        intName = 'lsoda'
    else:
        if minE >= -2:
            intName = 'dopri5'
        else:
            intName = 'vode'

    return intName


def vecToMatSens(s, numState, numParam):
    '''
    Convert the vector of :class:`numpy.ndarray` of forward
    sensitivities into a matrix.

    Parameters
    ----------
    s:
        forward sensitivities
    numState:
        number of state for the ode
    numParam:
        the number of parameters

    Returns
    -------
    S in matrix form

    See also
    --------
    :func:`matToVecSens`

    '''
    return np.reshape(s, (numState, numParam), 'F')

def vecToMatFF(ff, numState, numParam):
    '''
    Convert the vector of :class:`numpy.ndarray` of forward
    forward sensitivities into a matrix.

    Parameters
    ----------
    ff:
        forward forward sensitivities
    numState:
        number of state for the ode
    numParam:
        the number of parameters

    Returns
    -------
    FF in matrix form

    See also
    --------
    :func:`matToVecFF`

    '''
    return np.reshape(ff, (numState*numParam, numParam))

def matToVecSens(S, numState, numParam):
    '''
    Convert the matrix of :class:`numpy.ndarray` of forward
    sensitivities into a vector.

    Parameters
    ----------
    S:
        forward sensitivities
    numState:
        number of state for the ode
    numParam:
        the number of parameters

    Returns
    -------
    s in vector form

    See also
    --------
    :func:`vecToMatSens`

    '''
    # this can also be
    # S.flatten('F')
    # not sure which one is better
    return np.reshape(S, numState * numParam, order='F')

def matToVecFF(FF, numState, numParam):
    '''
    Convert the matrix of :class:`numpy.ndarray` of forward
    forward sensitivities into a vector.

    Parameters
    ----------
    FF:
        forward forward sensitivities
    numState:
        number of state for the ode
    numParam:
        the number of parameters

    Returns
    -------
    ff in vector form

    See also
    --------
    :func:`vecToMatFF`

    '''
    return FF.ravel()

class compileCode(object):
    '''
    A class that compiles an algebraic expression in sympy to a faster
    numerical file using the appropriate backend.
    '''

    def __init__(self, backend=None):
        '''
        Initializing the class.  Automatically checks which backend is
        available.  Currently only those linked to np are used where
        those linked with Theano are not.
        '''
        if backend is None:
            self._backend = None
            x = sympy.Symbol('x')
            expr = sympy.sin(x)/x

            # lets assume that we can't do theano.... (for now)
            # now lets explore the other options
            try:
                # first, f2py.  This is the best because Cython below may
                # throw out errors with older versions of sympy due to a
                # bug (calling np.h, a c header file which has problem
                # dealing with vector output).
                a = autowrap(expr, args=[x])
                a(1)
                # congrats!
                self._backend = 'f2py'
            except:
                try:
                    import cython
                    a = autowrap(expr, args=[x], backend='Cython')
                    a(1)
                    # also need to test the matrix version because
                    # previous version of sympy does not work when compiling
                    # a matrix
                    exprMatrix = sympy.zeros(2,1)
                    exprMatrix[0] = expr
                    exprMatrix[1] = expr

                    a = autowrap(exprMatrix, args=[x], backend='Cython')
                    a(1)

                    self._backend = 'Cython'
                except:
                    # we have truely failed in life.  A standard lambda function!
                    # unfortunately, this may be the case when we are running
                    # stuff in a parallel setting where we create objects in
                    # pure computation nodes with no compile mechanism
                    self._backend = 'lambda'
        else:
            self._backend = backend

    def compileExpr(self, inputSymb, inputExpr, backend=None, compileType=False):
        '''
        Compiles the expression given the symbols.  Determines the backend
        if required.

        Parameters
        ----------
        inputSymb: list
            the set of symbols for the input expression
        inputExpr: expr
            expression in sympy
        backend: optional
            the backend we want to use to compile
        compileType: optional
            defaults to False.  If True, return an extra output that informs
            the end user of the method used to compile the equation, can be
            one of (np, mpmath, sympy)

        Returns
        -------
        Compiled function taking arguments of the input symbols
        '''
        if backend is None:
            backend = self._backend

        # unless specified, we are always going to use np and forget
        # about the floating point importance
        compiledFunc = None
        compileTypeChosen = None
        try:
            if backend == 'f2py':
                compiledFunc = autowrap(expr=inputExpr,
                                        args=inputSymb,
                                        backend='f2py')
                compileTypeChosen = 'np'
            elif backend == 'lambda':
                compiledFunc = lambdify(expr=inputExpr,
                                        args=inputSymb,
                                        modules='np')
                compileTypeChosen = 'np'
            elif backend.lower() in ('cython', 'np'):
                # note that we have another test layer because of the
                # bug previously mentioned in __init__ of this class
                try:
                    compiledFunc = autowrap(expr=inputExpr,
                                            args=inputSymb,
                                            backend='Cython')
                    compileTypeChosen = 'np'
                except:
                    # although we don't think it is possible given the checks
                    # previously performed, we should still try it
                    try:
                        compiledFunc = autowrap(expr=inputExpr,
                                                args=inputSymb,
                                                backend='f2py')
                        compileTypeChosen = 'np'
                    except:
                        compiledFunc = lambdify(expr=inputExpr,
                                                args=inputSymb,
                                                modules='np')
                        compileTypeChosen = 'np'
            else:
                raise ExpressionErrror("The problem is too tough")
        except:
            try:
                compiledFunc = lambdify(expr=inputExpr,
                                        args=inputSymb,
                                        modules='mpmath')
                compileTypeChosen = 'mpmath'
            except:
                compiledFunc = lambdify(expr=inputExpr,
                                        args=inputSymb,
                                        modules='sympy')
                compileTypeChosen = 'sympy'

        logging.debug('Compiled expression as {}'.format(compileTypeChosen))
        if compileType:
            return compiledFunc, compileTypeChosen
        else:
            return compiledFunc

    def compileExprAndFormat(self, inputSymb, inputExpr,
                             backend=None, modules=None, outType=None):
        '''
        Compiles the expression given the symbols and determine which
        type of output is it.  Transforms the output appropriately into
        numpy

        Parameters
        ----------
        inputSymb: list
            the set of symbols for the input expression
        inputExpr: expr
            expression in sympy
        backend: optional
            the backend we want to use to compile
        modules: optional
            in the event that f2py and Cython fails, which modules
            do we want to try and compile against

        Returns
        -------
        Function determined from the input using closures.
        '''

        a, compileType = self.compileExpr(inputSymb, inputExpr, backend, True)
        numRow = inputExpr.rows
        numCol = inputExpr.cols

        # define the different types of output
        if outType is None:
            if numRow == 1 or numCol == 1:
                outType = "vec"
            else:
                outType = "mat"

        if outType.lower() == "vec":
            if compileType == 'np':
                return lambda x: a(*x).ravel()
            else:
                return lambda x: np.array(a(*x).tolist(), float).ravel()
        elif outType.lower() == "mat":
            if compileType == 'np':
                return lambda x: a(*x)
            else:
                return lambda x: np.array(a(*x).tolist(), float)
        else:
            raise RuntimeError("Specified type of output not recognized")

