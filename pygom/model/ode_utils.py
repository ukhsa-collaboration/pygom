'''
Utilities used throughout the package.
'''

__all__ = [
    'shapeAdjust',
    'integrate',
    'integrateFuncJac',
    'compileCode'
    ]

from ._modelErrors import ArrayError, ExpressionErrror, InputError, IntegrationError

import numpy
import math
import matplotlib.pyplot
import scipy.sparse, scipy.integrate
import sympy
# from sympy.printing.theanocode import theano_function
# from sympy.utilities.autowrap import ufuncify
from sympy.utilities.lambdify import lambdify
from sympy.utilities.autowrap import autowrap

atol = 1e-10
rtol = 1e-10

class shapeAdjust(object):
    '''
    A class that change vector into matrix and vice versa for
    vectors used in :class:`OperateOdeModel`

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
            If True, then returns I kron A.  If False then A kron I
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
            If True, then returns I kron A.  If False then A kron I
        '''
        if pre:
            return scipy.sparse.kron(scipy.sparse.eye(self._p), A)
        else:
            return scipy.sparse.kron(A, scipy.sparse.eye(self._p))

def integrate(ode, x0, t, full_output=False):
    '''
    A wrapper on top of :mod:`odeint <scipy.integrate.odeint>` using
    :class:`OperateOdeModel <pygom.model.OperateOdeModel>`.

    Parameters
    ----------
    ode: object
        of type :class:`OperateOdeModel <pygom.model.OperateOdeModel>`
    t: array like
        the time points including initial time
    full_output: bool, optional
        If the additional information from the integration is required

    '''

    # INTEGRATE!!! (shout it out loud, in Dalek voice)
    # determine the number of output we want
    solution, output = scipy.integrate.odeint(ode.ode,
                                              x0, t,
                                              Dfun=ode.Jacobian,
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
                     full_output=False, intName=None, nsteps=10000):
    '''
    A replacement for :mod:`scipy.integrate.odeint` which performs integration using
    :class:`scipy.integrate.ode`, tries to pick the correct integration method at
    the start through eigenvalue analysis

    Parameters
    ----------
    func: callable
        the ode f(x)
    jac: callable
        Jacobian of the ode, :math:`J_{i,j} = \\nabla_{x_{j} f_{i}(x)
    x0: float
        initial value of the states
    t0: float
        initial time
    args: tuple, optional
        additional arguments to be passed on
    includeOrigin: bool, optional
        if the output should include the initial states x0
    full_output: bool, optional
        if additional output is required
    intName: str, optional
        the integration method.  All those availble in :class:`ode <scipy.integrate.ode>`
        are allowed with 'vode' and 'ivode' representing the non-stiff and stiff version
        respectively.  Defaults to None, which tries to choose the integration method
        via eigenvalue analysis (only one) using the initial conditions
    nstep: int, optional
        number of steps allowed between each time point of the integration

    Returns
    -------
    solution: array like
        a :class:`numpy.ndarray` of shape (len(t),len(x0)) if includeOrigin is False, else
        an extra row with x0 being the first.
    output : dict, only returned if full_output == True
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
    if intName is None:
        if full_output==True:
            # obtain the eigenvalue
            e = numpy.linalg.eig(jac(t0, x0, *args))[0]
            intName = _determineIntegratorGivenEigenValue(e)

            # the min and max of them
            # maxE = max(e)
            # minE = min(e)
            # if maxE >= 0:
            #     intName = 'lsoda'
            # else:
            #     if minE >= -2:
            #         intName = 'dopri5'
            #     else:
            #         intName = 'vode'
        else:
            intName = 'lsoda'

    r = _setupIntegrator(func, jac, x0, t0, args, intName, nsteps)
    # print intName
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

    if isNumeric(t):
        # force it to something iterable
        t = [t]
    elif isListLike(t): #, (numpy.ndarray, list, tuple)):
        pass
    else:  #
        raise InputError("Type of input time is not of a recognized type")

    for deltaT in t:
        if full_output:
            o1, o2, o3, o4, o5 = _integrateOneStep(r, deltaT, func, jac, args, True)
            successInfo.append(o2)
            eigenInfo.append(o3)
            maxEigen.append(o4)
            minEigen.append(o5)
            intName = _determineIntegratorGivenEigenValue(o3)            
            r = _setupIntegrator(func, jac, o1, deltaT, args, intName, nsteps)
        else:
            # TODO: switches
            o1 = _integrateOneStep(r, deltaT, func, jac, args, False)
        # append solution, same thing whether the output is full or not
        solution.append(o1)
    # finish integration

    solution = numpy.array(solution)
    
    if full_output == True:
        # have both
        output = dict()
        output['ev'] = numpy.array(eigenInfo)
        output['minev'] = numpy.array(minEigen)
        output['maxev'] = numpy.array(maxEigen)
        output['suc'] = numpy.array(successInfo)
        output['in'] = intName
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
            e = numpy.linalg.eig(jac(r.t, r.y, *args))[0]
            return r.y, r.successful(), e, max(e), min(e)
        else:
            return r.y
    else:
        try:
            numpy.linalg.eig(jac(r.t, r.y, *args))
        except:
            raise IntegrationError("Failed integration with x =  " + str(r.y))
        else:
            a = numpy.linalg.eig(jac(r.t, r.y, *args))[0]
            raise IntegrationError("Failed integration with x =  " + str(r.y) +
                                   " and max/min eigenvalues of the Jacobian is "
                                   + str(max(a)) + " and " + str(min(a)))

def _setupIntegrator(func, jac, x0, t0, args=(), intName=None, nsteps=10000):
    if intName == 'dopri5':
        # if we are going to use rk5, then one thing for sure is that we know for sure
        # that the set of equations are not stiff.  Furthermore, the Jacobian information
        # will never be used as it evaluate f(x) directly
        r = scipy.integrate.ode(func).set_integrator('dopri5', nsteps=nsteps,
                                                     atol=atol, rtol=rtol)
    elif intName == 'dop853':
        r = scipy.integrate.ode(func).set_integrator('dop853', nsteps=nsteps,
                                                     atol=atol, rtol=rtol)
    elif intName == 'vode':
        r = scipy.integrate.ode(func, jac).set_integrator('vode', with_jacobian=True,
                                                          lband=None, uband=None,
                                                          nsteps=nsteps, atol=atol, rtol=rtol)
    elif intName == 'ivode':
        r = scipy.integrate.ode(func, jac).set_integrator('vode', method='bdf', with_jacobian=True,
                                                          lband=None, uband=None,
                                                          nsteps=nsteps, atol=atol, rtol=rtol)
    elif intName == 'lsoda':
        r = scipy.integrate.ode(func, jac).set_integrator('lsoda', with_jacobian=True,
                                                          lband=None, uband=None, nsteps=nsteps,
                                                          atol=atol, rtol=rtol)
    else:
        r = scipy.integrate.ode(func, jac).set_integrator('lsoda', with_jacobian=True,
                                                          lband=None, uband=None, nsteps=nsteps,
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

def plot(solution, t, stateList=None, y=None, yStateList=None):
    '''
    Plot the results of the integration

    Parameters
    ==========
    solution: :class:`numpy.ndarray`
        solution from the integration
    t: array like
        the vector of time where the integration output correspond to
    stateList: list
        name of the states, if available

    Notes
     -----
    If we have 5 states or more, it will always be arrange such
    that it has 3 columns.
    '''

    assert isinstance(solution, numpy.ndarray), "Expecting an numpy.ndarray"
    # if not isinstance(solution, numpy.ndarray):
    #     raise InputError("Expecting an numpy.ndarray")

    # tests on solution
    if len(solution) == solution.size:
        numState = 1
    else:
        numState = len(solution[0, :])

    assert len(solution)==len(t), "Number of solution not equal to t"
    # if len(solution) != len(t):
    #     raise InputError("Number of solution not equal to t")

    if stateList is not None:
        if len(stateList) != numState:
            raise InputError("Number of state (string) should be equal to number of output")
        stateList = [str(i) for i in stateList]

    # tests for y
    if y is not None:
        y = checkArrayType(y)
        # if type(y) != numpy.ndarray:
        #     y = numpy.array(y)

        numTargetSol = len(y)
        # we test the validity of the input first
        if numTargetSol != len(t):
            raise InputError("Number of realization of y not equal to t")
        # then obtain the information
        if y.size == numTargetSol:
            numTargetState = 1
            y = y.reshape((numTargetSol, 1))
        else:
            numTargetState = y.shape[1]

        if yStateList is None:
            if numTargetState != numState:
                if stateList is None:
                    raise InputError("Unable to identify which observations the states"
                                    + " belong to")
                else:
                    nonAuto = False
                    for i in stateList:
                        # we are assuming here that we always name our
                        # time state as \tau when it is a non-autonomous system
                        if str(i) == 'tau':
                            nonAuto = True

                    if nonAuto == True:
                        if y.shape[1] != (solution.shape[1] - 1):
                            raise InputError("Size of y not equal to yhat")
                        else:
                            yStateList = list()
                            # we assume that our observation y follows the same
                            # sequence as the states and copy over without the
                            # time component
                            for i in stateList:
                                # test
                                if str(i) != 'tau':
                                    yStateList.append(str(i))
                    else:
                        raise InputError("Size of y not equal to yhat")
            else:
                yStateList = stateList
        else:
            if numTargetState == 1:
                if yStateList in (tuple, list):
                    if len(yStateList) != numTargetState:
                        raise InputError("Number of target state not equal to y")
                    else:
                        yStateList = [str(i) for i in yStateList]
                else:
                    if isinstance(yStateList, str):
                        yStateList = [yStateList]
                    elif isinstance(yStateList, sympy.Symbol):
                        yStateList = [str(yStateList)]
                    elif isinstance(yStateList, list):
                        assert len(yStateList) == 1, "Only have one target state"
                    else:
                        raise InputError("Not recognized input for yStateList")
            else:
                if numTargetState > numState:
                    raise InputError("Number of target state cannot be larger"
                                    + " than the number of state")

    # # let's take a moment and appreciate that we have finished checking

    # note that we can probably reduce the codes here significantly but
    # i have not thought of a good way of doing it yet.
    if numState > 9:
        numFigure = int(math.ceil(numState / 9.0))
        k = 0
        last = False
        # loop over all the figures minus 1
        for z in range(0, numFigure - 1):
            f, axarr = matplotlib.pyplot.subplots(3, 3)
            for i in range(0, 3):
                for j in range(0, 3):
                    axarr[i, j].plot(t, solution[:, k])
                    if stateList is not None:
                        axarr[i, j].set_title(stateList[k])
                        if yStateList is not None:
                            if stateList[k] in yStateList:
                                axarr[i, j].plot(t, y[:, yStateList.index(stateList[k])], 'r')
                        axarr[i, j].set_xlabel('Time')
                    k += 1
            # a single plot finished, now we move on to the next one

        # now we are getting to the last one
        row = int(math.ceil((numState - (9 * (numFigure - 1))) / 3.0))
        f, axarr = matplotlib.pyplot.subplots(row, 3)
        if row == 1:
            for j in range(0, 3):
                if last == True:
                    break
                axarr[j].plot(t, solution[:, k])
                if stateList is not None:
                    axarr[j].set_title(stateList[k])
                    if yStateList is not None:
                        if stateList[k] in yStateList:
                            axarr[j].plot(t, y[:, yStateList.index(stateList[k])], 'r')
                    axarr[j].set_xlabel('Time')
                k += 1
                if k == numState:
                    last = True
        else:
            for i in range(0, row):
                if last == True:
                    break
                for j in range(0, 3):
                    if last == True:
                        break
                    axarr[i, j].plot(t, solution[:, k])
                    if stateList is not None:
                        axarr[i, j].set_title(stateList[k])
                        if yStateList is not None:
                            if stateList[k] in yStateList:
                                axarr[i, j].plot(t, y[:, yStateList.index(stateList[k])], 'r')
                        axarr[i, j].set_xlabel('Time')
                    k += 1
                    if k == numState:
                        last = True

    elif numState <= 3:
        if numState == 1:
            # we only have one state, easy stuff
            f, axarr = matplotlib.pyplot.subplots(1, 1)
            matplotlib.pyplot.plot(t, solution)
            if stateList is not None:
                matplotlib.pyplot.plot(stateList[0])
        else:
            # we can deal with it in a single plot, in the format of 1x3
            f, axarr = matplotlib.pyplot.subplots(1, numState)
            for i in range(0, numState):
                axarr[i].plot(t, solution[:, i])
                if stateList is not None:
                    axarr[i].set_title(stateList[i])
                    if yStateList is not None:
                        if stateList[i] in yStateList:
                            yStateList.index(stateList[i])
                            axarr[i].plot(t, y[:, yStateList.index(stateList[i])], 'r')
                    # label :)
                    axarr[i].set_xlabel('Time')

    elif numState == 4:
        # we have a total of 4 plots, nice and easy display of a 2x2.  Going across
        # first before going down
        f, axarr = matplotlib.pyplot.subplots(2, 2)
        k = 0
        for i in range(0, 2):
            for j in range(0, 2):
                axarr[i, j].plot(t, solution[:, k])
                if stateList is not None:
                    axarr[i, j].set_title(stateList[k])
                    if yStateList is not None:
                        if stateList[k] in yStateList:
                            axarr[i, j].plot(t, y[:, yStateList.index(stateList[k])], 'r')
                    # label :)
                    axarr[i, j].set_xlabel('Time')
                k += 1
                if numState == k:
                    break
    else:
        row = int(math.ceil(numState / 3.0))
        # print(row)
        f, axarr = matplotlib.pyplot.subplots(row, 3)
        k = 0
        for i in range(0, row):
            for j in range(0, 3):
                axarr[i, j].plot(t, solution[:, k])
                if stateList is not None:
                    axarr[i, j].set_title(stateList[k])
                    if yStateList is not None:
                        if stateList[k] in yStateList:
                            axarr[i, j].plot(t, y[:, yStateList.index(stateList[k])], 'r')
                    axarr[i, j].set_xlabel('Time')
                k += 1
                if numState == k:
                    break
    # finish all options, now we have plotted.
    # tidy up the output.  Without tight_layout() we will have
    # numbers in the axis overlapping each other (potentially)
    f.tight_layout()
    matplotlib.pyplot.show()

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
    return numpy.reshape(s, (numState, numParam), 'F')

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
    return numpy.reshape(ff, (numState * numParam, numParam))

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
    return numpy.reshape(S, numState * numParam, order='F')

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
        available.  Currently only those linked to numpy are used where
        those linked with Theano are not.
        '''
        if backend is None:
            self._backend = None
            x = sympy.Symbol('x')
            expr = sympy.sin(x) / x

            # lets assume that we can't do theano.... (for now)
            # now lets explore the other options
            try:
                # first, f2py.  This is the best because Cython below may
                # throw out errors with older versions of sympy due to a
                # bug (calling numpy.h, a c header file which has problem
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

    def compileExpr(self, inputSymb, inputExpr, backend=None, modules=None):
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
        modules: optional
            in the event that f2py and Cython fails, which modules
            do we want to try and compile against

        Returns
        -------
        Compiled function taking arguments of the input symbols
        '''
        if backend is None:
            backend = self._backend

        # unless specified, we are always going to use numpy and forget
        # about the floating point importance
        try:
            if modules is None:
                modules = ['numpy', 'mpmath', 'sympy']
                # we would probably always want to use Theano first if possible
                if self._backend == 'f2py':
                    return autowrap(expr=inputExpr, args=inputSymb, backend='f2py')
                elif self._backend == 'lambda':
                    return lambdify(expr=inputExpr, args=inputSymb, modules=modules)
                elif self._backend == 'Cython':
                    # note that we have another test layer because of the bug previously
                    # mentioned in __init__ of this class
                    try:
                        return autowrap(expr=inputExpr, args=inputSymb, backend='Cython')
                    except:
                        # although we don't think it is possible given the checks
                        # previously performed, we should still try it
                        try:
                            return autowrap(expr=inputExpr, args=inputSymb, backend='f2py')
                        except:
                            return lambdify(expr=inputExpr, args=inputSymb, modules=modules)
                else:
                    raise ExpressionErrror("The problem is too tough")
            else:
                # we have module input, we assume that we want to compile the function
                # according to the input
                return lambdify(expr=inputExpr, args=inputSymb, modules=modules)
        except:
            try:
                return lambdify(expr=inputExpr, args=inputSymb, modules='mpmath')
            except:
                return lambdify(expr=inputExpr, args=inputSymb, modules='sympy')

    def compileExprAndFormat(self, inputSymb, inputExpr, backend=None, modules=None, outType=None):
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

        a = self.compileExpr(inputSymb, inputExpr, backend, modules)
        numRow = inputExpr.rows
        numCol = inputExpr.cols
        numIn = len(inputSymb)
        # evaluate the compiled function
        # note that it may return rubbish, with overflow, underflow
        # and division by zero.  We do not care about that, we only
        # want to know the type of output
        b = a(*numpy.ones(numIn))

        # define the different types of output

        # applicable when the output is already an numpy.ndarray
        # Defining a set of closures
        def outVec1(y): return a(*y).ravel()
        def outMat1(y): return a(*y)
        # if the output is matrix
        def outVec2(y): return numpy.asarray(a(*y)).ravel()
        def outMat2(y): return numpy.asarray(a(*y))
        # if it is something unknown, i.e. mpmath objects
        def outVec3(y): return numpy.array(a(*y).tolist(), float).ravel()
        def outMat3(y): return numpy.array(a(*y).tolist(), float)

        if outType is None:
            # now we test the type of output we got from evaluating the
            # compiled function to determine the suitable adjustment
            if type(b) == numpy.ndarray:
                if numRow == 1 or numCol == 1:
                    return outVec1
                else:
                    return outMat1
            elif type(b) == numpy.matrixlib.defmatrix.matrix:
                if numRow == 1 or numCol == 1:
                    return outVec2
                else:
                    return outMat2
            else:
                if numRow == 1 or numCol == 1:
                    return outVec3
                else:
                    return outMat3
        else:
            if outType.lower() == "vec":
                if type(b) == numpy.ndarray:
                    return outVec1
                elif type(b) == numpy.matrixlib.defmatrix.matrix:
                    return outVec2
                else:
                    return outVec3
            elif outType.lower() == "mat":
                if type(b) == numpy.ndarray:
                    return outMat1
                elif type(b) == numpy.matrixlib.defmatrix.matrix:
                    return outMat2
                else:
                    return outMat3
            else:
                raise RuntimeError("Specified type of output not recognized")

def checkArrayType(x):
    '''
    Check to see if the type of input is suitable.  Only operate on one
    or two dimension arrays

    Parameters
    ----------
    x: array like
        which can be either a :class:`numpy.ndarray` or list or tuple

    Returns
    -------
    x: :class:`numpy.ndarray`
        checked and converted array
    '''

    if isinstance(x, numpy.ndarray):
        pass
    elif isinstance(x, (list, tuple)):
        if isNumeric(x[0]):
            x = numpy.array(x)
        elif isinstance(x[0], (list, tuple, numpy.ndarray)):
            if isNumeric(x[0][0]):
                x = numpy.array(x)
            else:
                raise ArrayError("Expecting elements of float or int")
        else:
            raise ArrayError("Expecting elements of float or int")
    elif isNumeric(x):
        x = numpy.array([x])
    else:
        raise ArrayError("Expecting an array like object")

    return x

def checkDimension(x, y):
    '''
    Compare the length of two array like objects.  Converting both to a numpy
    array in the process if they are not already one.

    Parameters
    ----------
    x: array like
        first array
    y: array like
        second array

    Returns
    -------
    x: :class:`numpy.array`
        checked and converted first array
    y: :class:`numpy.array`
        checked and converted second array
    '''

    y = checkArrayType(y)
    x = checkArrayType(x)

    if len(y) != len(x):
        raise InputError("The number of observations and time points should have the same length")

    return (x, y)


def isNumeric(x):
    '''
    Test whether the input is a numeric

    Parameters
    ----------
    x: 
        anything

    Returns
    -------
    bool:
        True if it belongs to one of the recognized data type from
        the list (int, numpy.int16, numpy.int32, numpy.int64,
        float, numpy.float16, numpy.float32, numpy.float64)
    '''
    return isinstance(x,
                      (int, numpy.int16, numpy.int32, numpy.int64,
                      float, numpy.float16, numpy.float32, numpy.float64))

def isListLike(x):
    '''
    Test whether the input is a type that behaves like a list, such
    as (list,tuple,numpy.ndarray)

    Parameters
    ----------
    x: 
        anything

    Returns
    -------
    bool:
        True if it belongs to one of the three expected type
        (list,tuple,numpy.ndarray)
    '''
    return isinstance(x, (list, tuple, numpy.ndarray))

def strOrList(x):
    '''
    Test to see whether input is a string or a list.  If it
    is a string, then we convert it to a list.
    
    Parameters
    ----------
    x:
        str or list
    
    Returns
    -------
    x:
        x in list form

    '''
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    elif isinstance(x, str):
        return [x]
    else:
        raise InputError("Expecting a string or list")

