from functools import partial

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps
from scipy.optimize import leastsq, minimize_scalar

def get_init(y, t, ode, theta=None, full_output=False):
    '''
    Get an initial guess of theta given the observations y and the
    corresponding time points t.

    Parameters
    ----------
    y: :array like
        observed values
    t: array like
        time
    ode: :class:`.DeterministicOde`
        an ode object
    theta: array like
        parameter value
    full_output: bool, optional
        True if the optimization result should be returned. Defaults to False.

    Returns
    -------
    theta: array like
        a guess of the parameters

    '''
    if theta is None:
        p = ode.num_param
        theta = np.ones(p)/2.0

    f = partial(_fitGivenSmoothness, y, t, ode, theta)
    output = minimize_scalar(f, bounds=(0,10), method='bounded')
    thetaNew = np.array(ode._paramValue)

    if full_output:
        return thetaNew, output
    else:
        return thetaNew

def _fitGivenSmoothness(y, t, ode, theta, s):
    # p = ode.getNumParam()
    # d = ode.getNumState()

    splineList = interpolate(y, t, s=s)
    interval = np.linspace(t[1], t[-1], 1000)
    # xApprox, fxApprox, t = _getApprox(splineList, interval)

    # g2 = partial(residual_sample, ode, fxApprox, xApprox, interval)
    # g2J = partial(jac_sample, ode, fxApprox, xApprox, interval)
    g2 = partial(residual_interpolant, ode, splineList, interval)
    g2J = partial(jac_interpolant, ode, splineList, interval)

    res = leastsq(func=g2, x0=theta, Dfun=g2J, full_output=True)

    loss = 0
    for spline in splineList:
        loss += spline.get_residual()
    # approximate the integral using fixed points
    r = np.reshape(res[2]['fvec']**2, (len(interval), len(splineList)), 'F')

    return (r.sum())*(interval[1] - interval[0]) + loss

def interpolate(solution, t, s=0):
    '''
    Interpolate the solution of the ode given the time points
    and a suitable smoothing vector using univariate spline

    Parameters
    ----------
    solution: :class:`numpy.ndarray`
        f(t) of the ode with the rows correspond to time
    t: array like
        time
    s: smoothing scalar, optional
        greater or equal to zero

    Returns
    -------
    splineList: list
        of :class:`scipy.interpolate.UnivariateSpline`
    '''

    n, p = solution.shape
    assert len(t) == n, "Number of observations and time point not equal"
    # if isinstance(s, (np.ndarray, list, tuple)):
    if hasattr(s, '__iter__'):
        if len(s) == 1:
            assert s >= 0, "Smoothing factor must be non-negative"
            s = np.ones(p)*s
        else:
            assert len(s) == p, "Number of smoothing factor must be " + \
                "equal to input solution columns"
    else:
        assert s >= 0, "Smoothing factor must be non-negative"
        s = np.ones(p)*s

    splineList = [UnivariateSpline(t, solution[:,j], s=s[j]) for j in range(p)]

    return splineList

def cost_grad_interpolant(ode, spline_list, t, theta):
    '''
    Returns the cost (sum of squared residuals) and the gradient between the
    first derivative of the interpolant and the function of the ode

    Parameters
    ----------
    ode: :class:`.DeterministicOde`
        an ode object
    spline_list: list
        list of :class:`scipy.interpolate.UnivariateSpline`
    t: array like
        time
    theta: array list
        parameter value

    Returns
    -------
    cost: double
        sum of squared residuals
    g:
        gradient of the squared residuals
    '''

    xApprox, fxApprox, t = _getApprox(spline_list, t)
    g, r = grad_sample(ode, fxApprox, xApprox, t, theta, output_residual=True)
    return (r.ravel()**2).sum(), g

def cost_interpolant(ode, spline_list, t, theta, vec=True, aggregate=True):
    '''
    Returns the cost (sum of squared residuals) between the first
    derivative of the interpolant and the function of the ode

    Parameters
    ----------
    ode: :class:`.DeterministicOde`
        an ode object
    spline_list: list
        list of :class:`scipy.interpolate.UnivariateSpline`
    t: array like
        time
    theta: array list
        paramter value
    vec: bool, optional
        if the matrix should be flattened to be a vector
    aggregate: bool, optional
        sum the vector/matrix

    Returns
    -------
    cost: double
        sum of squared residuals
    '''

    xApprox, fxApprox, t = _getApprox(spline_list, t)
    return cost_sample(ode, fxApprox, xApprox, t, theta, vec, aggregate)

def residual_interpolant(ode, spline_list, t, theta, vec=True):
    '''
    Returns the residuals between the first derivative of the
    interpolant and the function of the ode

    Parameters
    ----------
    ode: :class:`.DeterministicOde`
        an ode object
    spline_list: list
        list of :class:`scipy.interpolate.UnivariateSpline`
    t: array like
        time
    theta: array list
        parameter value
    vec: bool, optional
        if the matrix should be flattened to be a vector
    aggregate: bool, optional
        sum the vector/matrix

    Returns
    -------
    r: array list
        the residuals
    '''

    xApprox, fxApprox, t = _getApprox(spline_list, t)
    return residual_sample(ode, fxApprox, xApprox, t, theta, vec)

def jac_interpolant(ode, spline_list, t, theta, vec=True): 
    xApprox, fxApprox, t = _getApprox(spline_list, t)
    return jac_sample(ode, fxApprox, xApprox, t, theta, vec)

def grad_interpolant(ode, spline_list, t, theta, outputResidual=False):
    xApprox, fxApprox, t = _getApprox(spline_list, t)
    return grad_sample(ode, fxApprox, xApprox, t, theta, outputResidual)

def cost_sample(ode, fxApprox, xApprox, t, theta, vec=True, aggregate=True):
    '''
    Returns the cost (sum of squared residuals) between the first
    derivative of the interpolant and the function of the ode using
    samples at time points t.

    Parameters
    ----------
    ode: :class:`.DeterministicOde`
        an ode object
    fxApprox: list
        list of approximated values for the first derivative
    xApprox: list
        list of approximated values for the states
    t: array like
        time
    theta: array list
        parameter value
    vec: bool, optional
        if the matrix should be flattened to be a vector.
    aggregate: bool/str, optional
        sum the vector/matrix.  If this is equals to 'int' then the Simpsons
        rule is applied to the samples.  Also changes the behaviour of vec,
        where True outputs a vector where the elements contain the values of
        the integrand on each of the dimensions of the ode.  False returns
        the sum of this vector, a scalar.

    Returns
    -------
    r: array list
        the cost or the residuals if vec is True

    See Also
    --------
    :func:`residual_sample`
    '''

    if isinstance(aggregate, str):
        c = residual_sample(ode, fxApprox, xApprox, t, theta, vec=False)**2
        if aggregate.lower() == 'int':
            integrand = simps(c, x=t, axis=0)
            if vec:
                return integrand
            else:
                return np.sum(integrand)
        else:
            raise RuntimeError("Aggregation method not recognized")

    elif isinstance(aggregate, bool):
        c = residual_sample(ode, fxApprox, xApprox, t, theta, vec)**2
        if aggregate:
            return c.sum(0)
        else:
            return c
    else:
        raise RuntimeError("Aggregation method not recognized")

def residual_sample(ode, fxApprox, xApprox, t, theta, vec=True):
    '''
    Returns the residuals between the first derivative of the
    interpolant and the function of the ode using samples at
    time points t.

    Parameters
    ----------
    ode: :class:`.DeterministicOde`
        an ode object
    fxApprox: list
        list of approximated values for the first derivative
    xApprox: list
        list of approximated values for the states
    t: array like
        time
    theta: array list
        parameter value
    vec: bool, optional
        if the matrix should be flattened to be a vector

    Returns
    -------
    r: array list
        the residuals

    See Also
    --------
    :func:`cost_sample`
    '''
    ode.parameters = theta
    fx = np.zeros(fxApprox.shape)
    for i, x in enumerate(xApprox):
        fx[i] = ode.ode(x, t[i])

    if vec:
        return (fxApprox - fx).flatten('F')
    else:
        return fxApprox - fx

def jac_sample(ode, fxApprox, xApprox, t, theta, vec=True):
    '''
    Returns the Jacobian of the objective value using the state
    values of the interpolant given samples at time points t. Note
    that the parameters taken here is chosen to be same as
    :func:`cost_sample` for convenience.

    Parameters
    ----------
    ode: :class:`.DeterministicOde`
        an ode object
    fxApprox: list
        list of approximated values for the first derivative
    xApprox: list
        list of approximated values for the states
    t: array like
        time
    theta: array list
        parameter value
    vec: bool, optional
        if the matrix should be flattened to be a vector

    Returns
    -------
    r: array list
        the residuals

    See Also
    --------
    :func:`cost_sample`
    '''
    ode.parameters = theta
    n = len(fxApprox)
    d = ode.num_state
    p = ode.num_param
    g = np.zeros((n, d, p))
    for i, x in enumerate(xApprox):
        g[i] = -2*ode.grad(x, t[i])

    if vec:
        return np.reshape(g.transpose(1,0,2), (n*d, p))
    else:
        return g

def grad_sample(ode, fxApprox, xApprox, t, theta,
                vec=False, output_residual=False):
    '''
    Returns the gradient of the objective value using the state
    values of the interpolant given samples at time points t. Note
    that the parameters taken here is chosen to be same as
    :func:`cost_sample` for convenience.

    Parameters
    ----------
    ode: :class:`.DeterministicOde`
        an ode object
    fxApprox: list
        list of approximated values for the first derivative
    xApprox: list
        list of approximated values for the states
    t: array like
        time
    theta: array list
        parameter value
    vec: bool, optional
        if the matrix should be flattened to be a vector
    output_residual: bool, optional
        if True, then the residuals will be returned as an
        additional argument

    Returns
    -------
    g: :class:`numpy.ndarray`
        gradient of the objective function

    See Also
    --------
    :func:`jac_sample`
    '''
    ode.parameters = theta
    r = residual_sample(ode, fxApprox, xApprox, t, theta, vec=False)

    g = np.zeros((len(fxApprox), ode.num_param))
    for i, x in enumerate(xApprox):
        g[i] = -2*ode.grad(x, t[i]).T.dot(r[i])

    if output_residual:
        return g.sum(0), r
    else:
        return g.sum(0)

def _getApprox(splineList, t):
    '''
    Returns the approximated values of the states and the function
    value of the ode given the interpolants and a series of time
    points.  The interpolants are expected to have been interpolated
    against observations of the states.

    Parameters
    ----------
    splinelist: list of :class:`scipy.interpolate.UnivariateSpline`
        the interpolating function of the state values
    t: array like
        time

    Returns
    -------
    x: :class:`numpy.ndarray`
        extrapolation of the function at t
    fx: :class:`numpy.ndarray`
        extrapolation of the first derivative at t
    t: :class:`numpy.ndarray`
        the inputted time

    See Also
    --------
    :func:`cost_sample`, :func:`jac_sample`
    '''
    if not hasattr(t, '__iter__'):
        t = np.array([t])

    n = len(t)
    m = len(splineList)
    xApprox = np.zeros((n,m))
    fxApprox = np.zeros((n,m))
    for j, spline in enumerate(splineList):
        xApprox[:,j] = spline(t)
        fxApprox[:,j] = spline.derivative()(t)

    return xApprox, fxApprox, t

# def integrateFunc(ode, splineList, t, theta):
#     return (residual_interpolant(ode, splineList, t, theta, vec=False)**2).sum(1)
