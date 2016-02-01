import functools

import numpy
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps
from scipy.optimize import leastsq, minimize_scalar

def getInit(y, t, ode, theta=None, full_output=False):
    if theta is None:
        p = ode.getNumParam()
        theta = numpy.ones(p)/2.0
    f = functools.partial(_fitGivenSmoothness, y, t, ode, theta)
    output = minimize_scalar(f, bounds=(0,10), method='bounded')
    thetaNew = numpy.array(ode._paramValue) 
    if full_output:
        return thetaNew, output
    else:
        return thetaNew
    
def _fitGivenSmoothness(y, t, ode, theta, s):
    # p = ode.getNumParam()
    # d = ode.getNumState()

    splineList = interpolate(y, t, s=s)
    interval = numpy.linspace(t[1],t[-1],1000)
    # xApprox, fxApprox, t = _getApprox(splineList, interval)

    # g2 = functools.partial(residualSample, ode, fxApprox, xApprox, interval)
    # g2J = functools.partial(jacSample, ode, fxApprox, xApprox, interval)
    g2 = functools.partial(residualInterpolant, ode, splineList, interval)
    g2J = functools.partial(jacInterpolant, ode, splineList, interval)

    res = leastsq(func=g2, x0=theta, Dfun=g2J, full_output=True)

    loss = 0
    for spline in splineList:
        loss += spline.get_residual()
    # approximate the integral using fixed points
    r = numpy.reshape(res[2]['fvec']**2, (len(interval), len(splineList)), 'F')

    return (r.sum())*(interval[1]-interval[0]) + loss

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
    if isinstance(s, (numpy.ndarray, list, tuple)):
        if len(s) == 1:
            assert s >= 0, "Smoothing factor must be non-negative"
            s = numpy.ones(p)*s
        else:
            assert len(s) == p, "Number of smoothing factor must be equal to input solution columns"
    else:
        assert s >= 0, "Smoothing factor must be non-negative"
        s = numpy.ones(p)*s
    
    splineList = [UnivariateSpline(t, solution[:,j], s=s[j]) for j in range(0, p)]
    
    return splineList
        
def costGradInterpolant(ode, splineList, t, theta):
    '''
    Returns the cost (sum of squared residuals) and the gradient
    between the first derivative of the interpolant and the
    function of the ode
    
    Parameters
    ----------
    ode: :class:`pygom.model.OperateOdeModel`
        an ode object
    splineList: list
        list of :class:`scipy.interpolate.UnivariateSpline`
    t: array like
        time
    theta: array list
        paramter value 
        
    Returns
    -------
    cost: double
        sum of squared residuals
    g:
        gradient of the squared residuals
    '''

    xApprox, fxApprox, t = _getApprox(splineList, t)
    g, r = gradSample(ode, fxApprox, xApprox, t, theta, outputResidual=True)
    return (r.ravel()**2).sum(), g

def costInterpolant(ode, splineList, t, theta, vec=True, aggregate=True):
    '''
    Returns the cost (sum of squared residuals) between the first 
    derivative of the interpolant and the function of the ode
    
    Parameters
    ----------
    ode: :class:`pygom.model.OperateOdeModel`
        an ode object
    splineList: list
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
    
    xApprox, fxApprox, t = _getApprox(splineList, t)
    return costSample(ode, fxApprox, xApprox, t, theta, vec, aggregate)

def residualInterpolant(ode, splineList, t, theta, vec=True):
    '''
    Returns the residuals between the first derivative of the 
    interpolant and the function of the ode
    
    Parameters
    ----------
    ode: :class:`pygom.model.OperateOdeModel`
        an ode object
    splineList: list
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
    r: array list
        the residuals
    '''
    
    xApprox, fxApprox, t = _getApprox(splineList, t)
    return residualSample(ode, fxApprox, xApprox, t, theta, vec)

def jacInterpolant(ode, splineList, t, theta, vec=True): 
    xApprox, fxApprox, t = _getApprox(splineList, t)
    return jacSample(ode, fxApprox, xApprox, t, theta, vec)

def gradInterpolant(ode, splineList, t, theta, outputResidual=False):
    xApprox, fxApprox, t = _getApprox(splineList, t)
    return gradSample(ode, fxApprox, xApprox, t, theta, outputResidual)

def costSample(ode, fxApprox, xApprox, t, theta, vec=True, aggregate=True):
    '''
    Returns the cost (sum of squared residuals) between the first
    derivative of the interpolant and the function of the ode using
    samples at time point t.
    
    Parameters
    ----------
    ode: :class:`pygom.model.OperateOdeModel`
        an ode object
    splineList: list
        list of :class:`scipy.interpolate.UnivariateSpline`
    t: array like
        time
    theta: array list
        paramter value
    vec: bool, optional
        if the matrix should be flattened to be a vector
    aggregate: bool/str, optional
        sum the vector/matrix.  If this is equals to 'int' then
        the Simpsons rule is applied to the samples.  Also changes
        the behaviour of vec, where True outputs a vector where
        the elements contain the values of the integrand on each
        of the dimensions of the ode.  False returns the sum of this
        vector, a scalar.
        
    Returns
    -------
    r: array list
        the residuals
    '''
    
    if isinstance(aggregate, str):
        c = residualSample(ode, fxApprox, xApprox, t, theta, vec=False)**2
        if aggregate.lower() == 'int':
            integrand = simps(c, x=t, axis=0)
            if vec:
                return integrand
            else: 
                return sum(integrand) 
        else:
            raise RuntimeError('Aggregation method not recognized')
    elif isinstance(aggregate, bool):
        c = residualSample(ode, fxApprox, xApprox, t, theta, vec)**2
        if aggregate:
            return c.sum(0)
        else:
            return c
    else:
        raise RuntimeError('Aggregation method not recognized')

def residualSample(ode, fxApprox, xApprox, t, theta, vec=True):
    ode = ode.setParameters(theta)
    fx = numpy.zeros(fxApprox.shape)
    for i, x in enumerate(xApprox):
        fx[i] = ode.ode(x, t[i])

    if vec:
        return (fxApprox-fx).flatten('F')
    else:
        return fxApprox-fx

def jacSample(ode, fxApprox, xApprox, t, theta, vec=True):
    ode = ode.setParameters(theta)
    n = len(fxApprox)
    d = ode.getNumState()
    p = ode.getNumParam()
    g = numpy.zeros((n, d, p))
    for i, x in enumerate(xApprox):
        g[i] = -2*ode.Grad(x,t[i])

    if vec:
        return numpy.reshape(g.transpose(1,0,2), (n*d, p))
    else:
        return g

def gradSample(ode, fxApprox, xApprox, t, theta, vec=False, outputResidual=False):
    ode = ode.setParameters(theta)
    r = residualSample(ode, fxApprox, xApprox, t, theta, vec=False)

    g = numpy.zeros((len(fxApprox), ode.getNumParam()))
    for i, x in enumerate(xApprox):
        g[i] = -2*ode.Grad(x,t[i]).T.dot(r[i])

    if outputResidual:
        return g.sum(0), r
    else:
        return g.sum(0)

def _getApprox(splineList, t):
    if not isinstance(t, (numpy.ndarray, list, tuple)):
        t = numpy.array([t])

    n = len(t)
    m = len(splineList)
    xApprox = numpy.zeros((n,m))
    fxApprox = numpy.zeros((n,m))
    for j, spline in enumerate(splineList):
        xApprox[:,j] = spline(t)
        fxApprox[:,j] = spline.derivative()(t)

    return xApprox, fxApprox, t

def integrateFunc(ode, splineList, t, theta):
    return (residualInterpolant(ode, splineList, t, theta, vec=False)**2).sum(1)