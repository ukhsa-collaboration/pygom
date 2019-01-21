"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module that is used to calculate the confidence interval
    given the estimated parameters

"""

__all__ = [
    'asymptotic',
    'profile',
    'bootstrap',
    'geometric'
]

import copy

import sympy
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import leastsq, minimize, root

from pygom.model._model_errors import EstimateError, InputError
from pygom.utilR.distn import qchisq, qnorm
from .ode_loss import NormalLoss, SquareLoss, PoissonLoss

def asymptotic(obj, alpha=0.05, theta=None, lb=None, ub=None):
    '''
    Finds the confidence interval at the :math:`\\alpha` level
    under the :math:`\\mathcal{X}^{2}` assumption for the
    likelihood

    Parameters
    ----------
    obj: ode object
        an object initialized from :class:`BaseLoss`
    alpha: numeric, optional
        confidence level, :math:`0 < \\alpha < 1`.  Defaults to 0.05.
    theta: array like, optional
        the MLE parameters.  Defaults to None which then theta will be
        inferred from the input obj
    lb: array like, optional
        expected lower bound
    ub: array like, optional
        expected upper bound

    Returns
    -------
    l: array like
        lower confidence interval
    u: array like
        upper confidence interval
    '''

    alpha, theta, lb, ub = _checkInput(obj, alpha, theta, lb, ub)

    H = obj.hessian(theta)
    if np.any(np.linalg.eig(H)[0] <= 0.0):
        H = obj.jtj(theta)
        ## H = obj.fisher_information(theta)

    I = np.linalg.inv(H)

    q = 0.5*qchisq(1 - alpha, df=1)
    xU = theta + np.sqrt(q*np.diag(I))
    xL = theta - np.sqrt(q*np.diag(I))

    return xL, xU

def bootstrap(obj, alpha=0.05, theta=None, lb=None, ub=None,
              iteration=0, full_output=False):
    '''
    Finds the confidence interval at the :math:`\\alpha` level
    via bootstrap

    Parameters
    ----------
    obj: ode object
        an object initialized from :class:`BaseLoss`
    alpha: numeric, optional
        confidence level, :math:`0 < \\alpha < 1`. Defaults to 0.05.
    theta: array like, optional
        the MLE parameters
    lb: array like, optional
        upper bound for the parameters
    ub: array like, optional
        lower bound for the parameters
    iteration: int, optional
        number of bootstrap samples, defaults to 0 which is interpreted as
        :math:`2n` where :math:`n` is the number of data points.
    full_output: bool
        if the full set of estimates is required.

    Returns
    -------
    l: array like
        lower confidence interval
    u: array like
        upper confidence interval
    '''
    alpha, theta, lb, ub = _checkInput(obj, alpha, theta, lb, ub)

    yhat = obj._getSolution(theta)
    if len(yhat.shape) > 1:
        if yhat.shape[1] == 1: yhat = yhat.flatten()

    r = obj.residual()
    p = len(theta)
    if len(r) == r.size:
        n, m = len(r), 1
    else:
        n, m = r.shape

    if iteration == 0:
        iteration = 2*n
    if iteration < 100:
        iteration = 100

    setTheta = np.zeros((iteration, p))

    for i in range(iteration):
        ## TODO: parallel
        obj2 = copy.deepcopy(obj)
        obj2._y = yhat.copy()
        if m > 1:
            for j in range(m):
                obj2._y[:, j] += np.random.choice(r[:, j], n)
        else:
            obj2._y += np.random.choice(r, n)

        obj2._lossObj = obj2._setLossType()

        try:
            xhatT = obj2.fit(theta, lb, ub)
        except Exception as e:
            print(e)
            print(theta)
            print(i)
            print(obj2.gradient(theta))
            obj2.plot()
            raise Exception("WTF")

        setTheta[i] = xhatT.copy()

    xLB, xUB = np.zeros(p), np.zeros(p)

    for j in range(p):
        s = np.sort(setTheta[:, j])
        xLB[j] = s[np.int(np.ceil((alpha/2.0)*iteration))]
        xUB[j] = s[np.int(np.ceil((1.0 - alpha/2.0)*iteration))]

    if full_output:
        return xLB, xUB, setTheta
    else:
        return xLB, xUB

def geometric(obj, alpha=0.05, theta=None,
              method='jtj', geometry='o',
              full_output=False):
    '''
    Finds the geometric confidence interval under profiling
    at the :math:`\\alpha` level the normal approximation

    Parameters
    ----------
    obj: ode object
        an object initialized from :class:`BaseLoss`
    alpha: numeric
        confidence level, :math:`0 < \\alpha < 1`
    theta: array like, optional
        the MLE parameters.  When None given, it tries to estimate the
        optimal using methods provided by obj
    method: string
        construction of the covariance matrix.  jtj is the :math:`J^{\\top}`
        where :math:`J` is the Jacobian of the ode.  'hessian' is the hessian
        of the ode while 'fisher' is the fisher information found by
        :math:`cov(\\nabla_{\\theta}\\mathcal{L})`.
    geometry: string
        the two types of geometry defined in [Moolgavkar1987]_. c geometry uses
        the covariance at the maximum likelihood estimate
        :math:`\\hat{\\theta}`, while the 'o' geometry is the covariance
        defined at point :math:`\\theta`.
    full_output: bool, optional
        If True then both the l_path and u_path will be outputted, else only
        the point estimates of l and u

    Returns
    -------
    l: array like
        lower confidence interval
    u: array like
        upper confidence interval
    l_path: list
        path from :math:`\\hat{\\theta}` to the lower :math:`1 - \\alpha/2`
        point for all parameters
    u_path: list
        same as l_path but for the upper confidence interval

    '''
    alpha, theta, lb, ub = _checkInput(obj, alpha, theta, None, None)

    p = len(theta)
    xU, xL = np.zeros(p), np.zeros(p)
    xLList, xUList = [], []

    for i in range(p):
        dfFunc = _geometricOde(obj, alpha, theta, i, method, geometry)
        solutionU = odeint(dfFunc, theta, np.linspace(0, 1, 101))
        solutionL = odeint(dfFunc, theta, np.linspace(0, -1, 101))

        if full_output:
            xUList.append(solutionU.copy())
            xLList.append(solutionL.copy())

        xU[i] = solutionU[-1][i]
        xL[i] = solutionL[-1][i]

    if full_output:
        return xL, xU, xLList, xUList
    else:
        return xL, xU

def _geometricOde(obj, alpha, xhat, i, method='jtj', geometry='o'):

    if method.lower() == 'jtj':
        cov = obj.jtj
    elif method.lower() == 'hessian':
        cov = obj.hessian
    elif method.lower() == 'fisher':
        cov = obj.fisher_information
    else:
        raise Exception("Input method not recognized")

    H0 = cov(xhat)
    p = len(xhat)
    setIndex = set(range(p))
    activeIndex = list(setIndex - set([i]))
    k = qnorm(1 - alpha/2)

    def F1(x, t):
        H = cov(x)
        tau = np.ones(p)
        dwdb = -np.linalg.lstsq(H[activeIndex][:,activeIndex],
                                H[i,activeIndex],
                                rcond=None)[0] #To silence the FutureWarning
        tau[activeIndex] = dwdb
        if geometry.lower() == 'c':
            g = tau.T.dot(H0).dot(tau)
        elif geometry.lower() == 'o':
            g = tau.T.dot(H).dot(tau)
        else:
            raise Exception("Input geometry not recognized")

        df = np.ones(p)
        df[i] = k/np.sqrt(g)
        df[activeIndex] = dwdb * df[i]
        return df

    return F1

def profile(obj, alpha, theta=None, lb=None, ub=None, full_output=False):
    '''
    Finds the profile confidence interval at the
    :math:`\\alpha` level under the :math:`\\mathcal{X}^{2}`
    assumption for the likelihood

    Parameters
    ----------
    obj: ode object
        an object initialized from :class:`BaseLoss`
    alpha: numeric
        confidence level, :math:`0 < \\alpha < 1`
    theta: array like, optional
        the MLE parameters.  When None given, it tries to estimate the
        optimal using methods provided by obj
    lb: array like, optional
        expected lower bound
    ub: array like, optional
        expected upper bound
    full_output: bool, optional
        if more output is desired

    Returns
    -------
    l: array like
        lower confidence interval
    u: array like
        upper confidence interval
    '''
    alpha, theta, lb, ub = _checkInput(obj, alpha, theta, lb, ub)

    p = len(theta)
    xU, xL = np.zeros(p), np.zeros(p)
    xLList, xUList = [], []

    for i in range(p):
        xhatL, xhatU = _profileGetInitialValues(theta, i, alpha, obj,
                                                lb=lb, ub=ub)

        # define our functions: objective, gradient, hessian, and the
        # approximation to the hessian via Jacobian
        funcF = _profileF(theta, i, alpha, obj)
        funcFgradient = _profileFgradient(theta, i, alpha, obj)
        funcFhessian = _profileFhessian(theta, i, alpha, obj)

        lbT = np.ones(p)*-np.Inf if lb is None else lb.copy()
        ubT = np.ones(p)*np.Inf if ub is None else ub.copy()

        ubT[i] = theta[i]

        try:
            xTempL, outL = _profileObtainViaNuisance(xhatL, theta, i, alpha,
                                                     obj, lbT, ubT,
                                                     obtainLB=True,
                                                     full_output=True)
        except EstimateError:
            xTempL, outL = _profileObtainAndVerifyBounds(funcF,
                                                         funcFgradient,
                                                         funcFhessian,
                                                         xhatL, lbT, ubT, True)

        ## re-adjust the bounds for the other side
        lbT = np.ones(p)*-np.Inf if lb is None else lb.copy()
        ubT = np.ones(p)*np.Inf if ub is None else ub.copy()

        lbT[i] = theta[i]

        try:
            xTempU, outU = _profileObtainViaNuisance(xhatU, theta, i, alpha,
                                                     obj, lbT, ubT,
                                                     obtainLB=False,
                                                     full_output=True)
        except EstimateError:
            xTempU, outU = _profileObtainAndVerifyBounds(funcF,
                                                         funcFgradient,
                                                         funcFhessian,
                                                         xhatU, lbT, ubT, True)
        ## now we have to store the values in one go.
        ## So we go L -> U -> U -> L and below is not a typo
        xLList.append(outL)
        xUList.append(outU)

        xU[i] = xTempU[i]
        xL[i] = xTempL[i]

    if full_output:
        return xL, xU, xLList, xUList
    else:
        return xL, xU

def _profileGetInitialValues(theta, i, alpha, obj, approx=True,
                             lb=None, ub=None):
    '''
    We would not use an approximation in general because if the input theta
    is an optimal value, then we would expect the Hessian to be a PSD matrix.
    '''
    p = len(theta)
    setIndex = set(range(p))

    H = obj.jtj(theta) if approx == True else obj.hessian(theta)
    # if approx:
    #     H = obj.jtj(theta)
    # else:
    #     H = obj.hessian(theta)

    activeIndex = list(setIndex - set([i]))
    tau = np.ones(p)
    dwdb = -np.linalg.lstsq(H[activeIndex][:, activeIndex],
                            H[i, activeIndex],
                            rcond=None)[0] #To silence the FutureWarning)[0]
    tau[activeIndex] = dwdb

    q = qchisq(1 - alpha, df=1)
    h = np.sqrt(q/(H[i, i] + (H[i, activeIndex].T).dot(dwdb)))

    # we only move a half step and not a full step as a more
    # conservative approach is less likely to have shoot out of bounds
    xhatU = theta + 0.5*h*tau
    xhatL = theta - 0.5*h*tau

    if lb is not None:
        for i, lb_i in enumerate(lb):
            if xhatL[i] <= lb_i: xhatL[i] = lb_i

    if ub is not None:
        for i, ub_i in enumerate(ub):
            if xhatU[i] >= ub_i: xhatU[i] = ub_i

    return xhatL, xhatU

def _profileOptimizeNuisance(theta, i, obj, lb, ub):
    '''
    Find the minimized nuisance parameters given the parameter of interest.

    Parameters
    ----------
    theta: array like, optional
        current parameters values
    i: int
        index of the parameter of interest
    obj: ode object
        an object initialized from :class:`BaseLoss`
    lb: array like, optional
        expected lower bound
    ub: array like, optional
        expected upper bound

    Returns
    -------
    s: array like
        optimal value
    '''
    p = len(theta)
    setIndex = set(range(p))
    activeIndex = list(setIndex - set([i]))

    if isinstance(obj, NormalLoss):
        lossF = NormalLoss
    elif isinstance(obj, SquareLoss):
        lossF = SquareLoss
    elif isinstance(obj, PoissonLoss):
        lossF = PoissonLoss
    else:
        raise Exception("Loss type not supported")

    if obj._targetParam is None:
        targetParam2 = obj._ode.param_list()
    else:
        targetParam2 = obj._targetParam

    # targetParam3 = list()
    # for i in activeIndex:
    #     targetParam3.append(targetParam2[i])
    targetParam3 = [targetParam2[i] for i in activeIndex]

    ode2 = copy.deepcopy(obj._ode)
    ode2.setParameters(theta)
    objSIR2 = lossF(copy.deepcopy(obj._theta[activeIndex]),
                    ode2,
                    copy.deepcopy(obj._x0),
                    copy.deepcopy(obj._t[0]),
                    copy.deepcopy(obj._t[1::]),
                    copy.deepcopy(obj._y),
                    copy.deepcopy(obj._stateName),
                    copy.deepcopy(obj._stateWeight),
                    targetParam3,
                    copy.deepcopy(obj._targetState))

    boundsT = np.reshape(np.append(lb, ub), (len(lb), 2), 'F')

    res = minimize(fun=objSIR2.cost, jac=objSIR2.gradient,
                   x0=obj._theta[activeIndex],
                   bounds=boundsT[activeIndex,:],
                   method='L-BFGS-B') # , callback=objSIR2.thetaCallBack)

    return res['x']

def _profileObtainViaNuisance(theta, xhat, i, alpha, obj, lb, ub,
                              obtainLB=True, full_output=False):
    '''
    Find the profile likelihood confidence interval by iteratively minimizing
    over the nuisance parameters first then minimizing the parameter of
    interest, rather than tackling them all at the same time.
    '''
    xhatT = theta.copy()
    funcF = _profileF(xhat, i, alpha, obj)
    funcG = _profileG(xhat, i, alpha, obj)

    lbT, ubT = lb.copy(), ub.copy()
    # ubT = ub.copy()

    p = len(theta)
    setIndex = set(range(p))
    activeIndex = list(setIndex - set([i]))

    # note that the bounds here needs to be reversed.
    if obtainLB:
        ubT[activeIndex] = xhat[activeIndex]
    else:
        lbT[activeIndex] = xhat[activeIndex]

    # define the corresponding objective function that minimizes the nuisance
    # parameters internally.
    def ABC1(beta):
        xhatT[i] = beta
        xhatT[activeIndex] = _profileOptimizeNuisance(xhatT, i, obj, lbT, ubT)
        return funcG(xhatT)[i]

    def ABCJac(beta):
        xhatT[0] = beta
        xhatT[activeIndex] = _profileOptimizeNuisance(xhatT, 0, obj, lb, ub)
        g = funcG(xhatT)
        return np.array([2*g[0]*obj.gradient()[0]])

    try:
        res = root(ABC1, xhatT[i])
    except Exception:
        raise EstimateError("Error in using the direct root finder")

    ## res1 = root(ABC1,xhatL[0],jac=ABCJac)
    ## res = scipy.optimize.minimize_scalar(ABC,bounds=(?,?))

    if res['success'] == True:
        if obtainLB:
            # if we want the lower bound, then the estimate should not
            # be higher than the MLE
            if obj._theta[i] >= xhat[i]:
                raise EstimateError("Estimate higher than MLE")
        else:
            if obj._theta[i] <= xhat[i]:
                raise EstimateError("Estimate lower than MLE")

        res['method'] = 'Nested Minimization'
        if full_output is True:
            return obj._theta.copy(), res
        else:
            return obj._theta.copy()
    else:
        raise EstimateError("Failure in estimation of the profile likelihood: "
                            + res['message'])

def _profileObtainAndVerify(f, df, x0, full_output=False):
    '''
    Find the solution of the profile likelihood and check
    that the algorithm has converged.
    '''
    x, cov, infodict, mesg, ier = leastsq(func=f, x0=x0, Dfun=df,
                                          maxfev=10000, full_output=True)

    if ier not in (1, 2, 3, 4):
        raise EstimateError("Failure in estimation of the profile likelihood: "
                            + mesg)

    if full_output:
        output = dict()
        output['cov'] = cov
        output['infodict'] = infodict
        output['mesg'] = mesg
        output['ier'] = ier
        return x, output
    else:
        return x

def _profileObtainAndVerifyBounds(f, df, ddf, x0, lb, ub, full_output=False):
    res = minimize(fun=f, jac=df, # hess=ddf,
                   x0=x0,
                   bounds=np.reshape(np.append(lb, ub), (len(lb),2), 'F'),
                   method='l-bfgs-b',
                   options={'maxiter':1000})

    if res["success"] == False:
        raise EstimateError("Failure in estimation of the profile " +
                            "likelihood: " + res['message'])
    else:
        res["method"] = "Direct Minimization"

    if full_output:
        return res['x'], res
    else:
        return res['x']

def _checkInput(obj, alpha, theta, lb, ub):

    if alpha is None:
        alpha = 0.05
    elif alpha > 1.0:
        raise InputError("Cannot have a confidence level higher than 1")
    elif alpha < 0.0:
        raise InputError("Cannot have a confidence level lower than 0")

    if lb is None or ub is None:
        if ub is None:
            ub = np.array([None]*len(theta))
        if lb is None:
            lb = np.array([None]*len(theta))
    else:
        if len(lb) != len(ub):
            raise InputError("Number of lower and upper bound must be equal")
        if len(lb) != len(theta):
            raise InputError("Number of box constraints must equal to the" +
                             " number of variables")

    if theta is None:
        if ub is not None and lb is not None:
            theta = obj.fit(lb + (ub - lb)/2, lb=lb, ub=ub)
        else:
            raise InputError("Expecting the estimated parameter when box" +
                             "constraints are not supplied")

    return alpha, theta, lb, ub

def _profileF(xhat, i, alpha, obj):
    c = obj.cost(xhat) + 0.5*qchisq(1 - alpha, df=1)
    def func(x):
        r = obj.gradient(x)
        r[i] = obj.cost(x) - c
        return (r**2).sum()

    return func

def _profileG(xhat, i, alpha, obj):
    c = obj.cost(xhat) + 0.5*qchisq(1 - alpha, df=1)
    def func(x):
        r = obj.gradient(x)
        r[i] = obj.cost(x) - c
        return r

    return func

def _profileGSecondOrderCorrection(xhat, i, alpha, obj, approx=True):
    '''
    Finds the correction term when approximating the gradient to
    second order [Venzon1988]_, i.e. :math:`\\delta^{\\top} D(\\theta) \\delta`
    in [Venzon1988]_.  If the system
    of non-linear equations is a, then we return :math:`a + s`
    instead of :math:`G^{-1}a`, i.e. we have incorporated the correction
    into the gradient

    Parameters
    ----------
    x: array like
        current value of the parameters
    xhat: array like
        parameters at MLE
    i: int
        our target variable
    alpha: numeric
        confidence level, between :math:`(0,1)`
    obj:
        ode object
    approx: bool, optional
        default is True.

    Returns
    -------
    g: array like
        corrected set of non-linear equations

    '''
    s = sympy.symbols('s')
    c = obj.cost(xhat) + 0.5*qchisq(1 - alpha, df=1)
    D0 = obj.hessian(xhat)

    def func(x):
        # first, we obtain all the necessary information.
        # We use the notation in the original paper.
        # so that G is the derivative of the systems of
        # equations and JTJ is D(\theta)
        if approx:
            H,output = obj.jtj(x, full_output=True)
        else:
            H, output = obj.hessian(x, full_output=True)

        g = output['grad']
        G = H.copy()
        G[i] = g
        lvector = g.copy()
        lvector[i] = obj.cost(x) - c

        # computing the inverse, even though it is less
        # accurate then doing a least squares, we are saving
        # a lot of computation time here
        invG = np.linalg.inv(G)
        v = invG.dot(lvector)

        sTemp = v + sympy.Matrix(invG[:,i])*s
        RHS = (sTemp.T*sympy.Matrix(H)*sTemp)[0]
        sRoots = sympy.solve(sympy.Eq(2*s, RHS), s)
        abc = sympy.lambdify((),sympy.Matrix(sRoots), 'np')
        sRootsReal = np.asarray(abc()).real
        rootsSize = sRootsReal.size

        if rootsSize > 0:
            distL = np.zeros(len(sRootsReal))
            for j in range(rootsSize):
                vTemp = v.copy() + sRootsReal[j]*invG[:,i]
                distL[j] = vTemp.T.dot(D0.dot(vTemp))
                # finish finding the distance
            index = distL.argmin()
            lvector[i] += sRootsReal[index]
            return lvector
        else:
            return lvector

    return func

def _profileH(xhat, i, alpha, obj, approx=True):
    def func(x):
        if approx:
            H, output = obj.jtj(x, full_output=True)
        else:
            H, output = obj.hessian(x, full_output=True)

        H[i] = output['grad']
        return H

    return func

def _profileFgradient(xhat, i, alpha, obj, approx=True):
    c = obj.cost(xhat) + 0.5*qchisq(1 - alpha, df=1)

    def func(x):
        if approx:
            H, output = obj.jtj(x, full_output=True)
        else:
            H, output = obj.hessian(x, full_output=True)

        g = output['grad']
        G = H.copy()
        G[i] = g.copy()
        lvector = g.copy()
        lvector[i] = obj.cost(x) - c
        # note that now G is the Jacobian of the objective function
        # so we need a transpose
        return G.T.dot(2*lvector)

    return func

def _profileFhessian(xhat, i, alpha, obj, approx=True):
    c = obj.cost(xhat) + 0.5*qchisq(1 - alpha, df=1)

    def func(x):
        if approx:
            H, output = obj.jtj(x, full_output=True)
        else:
            H, output = obj.hessian(x, full_output=True)

        g = output['grad']
        G = H.copy()
        G[i] = g.copy()
        lvector = g.copy()
        lvector[i] = obj.cost(x) - c
        A = 2*G.T.dot(G)
        if not approx:
            for s in lvector:
                A += s*H
        return A

#         if approx:
#             return 2*G.T.dot(G)
#         else:
#             A = 2*G.T.dot(G)
#             ## here we assume that only the second derivative is
#             ## significant.
#             for s in lvector:
#                 A += s*H
#             ## return np.diag(lvector).dot(H) + 2 * G.T.dot(G)
#             ## return 2 * G.T.dot(G)
#             return A

    return func



