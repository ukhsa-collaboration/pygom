.. _profile:

*******************************************
Confidence Interval of Estimated Parameters
*******************************************

After obtaining the *best* fit, it is natural to report both the point estimate and the confidence level at the :math:`\alpha` level.  The easiest way to do this is by invoking the normality argument and use Fisher information of the likelihood.  As explained previously at the bottom of :ref:`gradient`, we can find the Hessian, :math:`\mathbf{H}`, or the approximated Hessian for the estimated parameters.  The Cramer--Rao inequality, we know that

.. math::
    Var(\hat{\theta}) \ge \frac{1}{I(\theta)},

where :math:`I(\theta)` is the Fisher information, which is the Hessian subject to regularity condition.  Given the Hessian, computing the confidence intervals is trivial.  Note that this is also known as the asymptotic confidence interval where the normality comes from invoking the CLT.  There are other ways of obtaining a confidence intervals, we will the ones implemented in the package.  First, we will set up a SIR model as seen in :ref:`sir` which will be used throughout this page.

.. ipython::

    In [1]: from pygom import NormalLoss, common_models

    In [2]: from pygom.utilR import qchisq

    In [3]: import numpy

    In [4]: import scipy.integrate

    In [5]: import matplotlib.pyplot as plt

    In [6]: import copy

    In [7]: ode = common_models.SIR([('beta', 0.5), ('gamma', 1.0/3.0)])

and we assume that we only have observed realization from the :math:`R` compartment

.. ipython::

    In [1]: x0 = [1, 1.27e-6, 0]

    In [2]: t = numpy.linspace(0, 150, 100).astype('float64')

    In [3]: ode.initial_values = (x0, t[0])

    In [4]: solution = ode.integrate(t[1::])

    In [5]: theta = [0.2, 0.2]

    In [6]: targetState = ['R']

    In [7]: targetStateIndex = numpy.array(ode.get_state_index(targetState))

    In [8]: y = solution[1::,targetStateIndex] + numpy.random.normal(0, 0.01, (len(solution[1::,targetStateIndex]), 1))

    In [9]: yObv = y.copy()

    In [10]: objSIR = NormalLoss(theta, ode, x0, t[0], t[1::], y, targetState)

    In [11]: boxBounds = [(1e-8, 2.0), (1e-8, 2.0)]

    In [12]: boxBoundsArray = numpy.array(boxBounds)

    In [13]: xhat = objSIR.fit(theta, lb=boxBoundsArray[:,0], ub=boxBoundsArray[:,1])

Asymptotic
==========

When the estimate is obtained say, under a square loss or a normal assumption, the corresponding likelihood can be written down easily.  In such a case, likelihood ratio test under a Chi--squared distribution is

.. math::

    2 (\mathcal{L}(\hat{\boldsymbol{\theta}}) - \mathcal{L}(\boldsymbol{\theta})) \le \chi_{1 - \alpha}^{2}(k)

where :math:`1-\alpha` is the size of the confidence region and :math:`k` is the degree of freedom.  The corresponding asymptotic confidence interval for parameter :math:`j` can be derived as

.. math::

    \hat{\theta}_{j} \pm \sqrt{\chi_{1 - \alpha}^{2}(k) H_{i,i}}.

A pointwise confidence interval is obtained when :math:`k = 1`.  We assume in our package that a pointwise confidence interval is desired.  This can be obtained simply by

.. ipython::

    In [1]: from pygom import confidence_interval as ci

    In [2]: alpha = 0.05

    In [3]: xL, xU = ci.asymptotic(objSIR, alpha, xhat, lb=boxBoundsArray[:,0], ub=boxBoundsArray[:,1])

    In [4]: print(xL)

    In [5]: print(xU)

Note that the set of bounds here is only used for check the validity of :math:`\hat{\mathbf{x}}` and not used in the calculation of the confidence intervals.   Therefore the resulting output can be outside of the box constraints.

Profile Likelihood
==================

Another approach to calculate the confidence interval is to tackle one parameter at a time, treating the rest of them as nuisance parameters, hence the term *profile*.  Let :math:`\mathcal{L}(\boldsymbol{\theta})` be our log--likelihood with parameter :math:`\boldsymbol{\theta}`.  Element :math:`\theta_{j}` is our parameter of interest and :math:`\boldsymbol{\theta}_{-j}` represents the complement such that :math:`\boldsymbol{\theta} = \theta_{j} \cup \boldsymbol{\theta}_{-j}`.  For simply models such as linear regression with only regression coefficients :math:`\boldsymbol{\beta}`, then :math:`\boldsymbol{\theta} = \boldsymbol{\beta}`.

To shorten the notation, let

.. math:: \mathcal{L}(\boldsymbol{\theta}_{-j} \mid \theta_{j}) = \max \mathcal{L}(\boldsymbol{\theta}_{-j} \mid \theta_{j})
    :label: nuisanceOptim

which is the maxima of :math:`\boldsymbol{\theta}_{-j}` given :math:`\theta_{j}`.  :math:`\hat{\boldsymbol{\theta}}` denotes the MLE of the parameters as usual.  The profile--likelihood based confidence interval for :math:`\theta_{j}` is defined as

.. math::

    \theta_{j}^{U} &= \sup \left\{ \mathcal{L}(\hat{\boldsymbol{\theta}}) - \mathcal{L}(\boldsymbol{\theta} \mid \theta_{j}) \le \frac{1}{2} \chi_{1 - \alpha}^{2}(1) \right\} \\
    \theta_{j}^{L} &= \inf \left\{ \mathcal{L}(\hat{\boldsymbol{\theta}}) - \mathcal{L}(\boldsymbol{\theta} \mid \theta_{j}) \le \frac{1}{2} \chi_{1 - \alpha}^{2}(1) \right\}

where again we have made use of the normal approximation, but without imposing symmetry.  The set of equations above automatically implies that the interval width is :math:`\theta_{j}^{U} - \theta_{j}^{L}` and

.. math::

    \mathcal{L}(\hat{\boldsymbol{\theta}}) - \frac{1}{2} \chi_{1-\alpha}^{2}(1) - \mathcal{L}(\boldsymbol{\theta} \mid \theta_{j}) = 0.

As mentioned previously, :math:`\boldsymbol{\theta}_{-j}` is the maximizer of the nuisance parameters, which has a gradient of zero.  Combining this with the equation above yields a non--linear system of equations of size :math:`p`,

.. math:: g(\boldsymbol{\theta}) = \left[ \begin{array}{c} \mathcal{L}(\boldsymbol{\theta} \mid \theta_{j}) - c \\ \frac{\partial \mathcal{L}(\boldsymbol{\theta} \mid \theta_{j})}{\partial \boldsymbol{\theta}_{-j}} \end{array} \right] = 0
    :label: obj

where :math:`c = \mathcal{L}(\hat{\boldsymbol{\theta}}) + \frac{1}{2} \chi_{1-\alpha}^{2}(1)`.  Solving this set of system of equations only need simple Newton like steps, possibly with correction terms as per [Venzon1988]_.  We provide a function to obtain such estimate

.. ipython::
    :verbatim:

    In [1]: xLProfile, xUProfile, xLProfileList, xUProfileList = ci.profile(objSIR, alpha, xhat, lb=boxBoundsArray[:,0], ub=boxBoundsArray[:,1], full_output=True)

but unfortunately this is not accurate most of the time due to the complicated surface at locations not around :math:`\hat{\theta}`.  This is a common scenario for non--linear least square problems because the Hessian is not guaranteed to be a PSD everywhere.  Therefore, a safeguard is in place to obtain the :math:`\theta_{j}^{U},\theta_{j}^{L}` by iteratively by updating :math:`\theta_{j}` and find the solution to :eq:`nuisanceOptim`.

Furthermore, we also provide the functions necessary to obtain the estimates such as the four below.

.. ipython::

    In [1]: i = 0

    In [1]: funcF = ci._profileF(xhat, i, 0.05, objSIR)

    In [2]: funcG = ci._profileG(xhat, i, 0.05, objSIR)

    In [3]: funcGC = ci._profileGSecondOrderCorrection(xhat, i, alpha, objSIR)

    In [4]: funcH = ci._profileH(xhat, i, 0.05, objSIR)

Where :math:`i` is the index of the parameter of interest.  :func:`_profileF` is the squared norm of :eq:`obj`, which easy the optimization process for solvers which requires a converted form from system of equations to non-linear least squares.  :func:`_profileG` is the systems of equations :eq:`obj`, :func:`_profileH` is the derivative of :eq:`obj`

.. math::
    \nabla g(\boldsymbol{\theta}) = \left[ \begin{array}{c} \frac{\partial \mathcal{L}(\boldsymbol{\theta} \mid \theta_{j})}{\partial \theta_{j}} \\ \frac{\partial^{2} \mathcal{L}(\boldsymbol{\theta} \mid \theta_{j})}{\partial \boldsymbol{\beta}_{-j} \partial \theta_{j}} \end{array} \right]

and :func:`_profileGSecondOrderCorrection` has the second order correction [Venzon1988]_.

Geometric profile likelihood
============================

Due to the difficulty in obtain a profile likelihood via the standard Newton like steps, we also provide a way to generate a similar result using the geometric structure of the likelihood surface.  We follow the method in [Moolgavkar1987]_, which involves solving a set of differential equations

.. math::
    \frac{d\beta_{j}}{dt} &= k g^{-1/2} \\
    \frac{d\boldsymbol{\beta}_{-j}}{dt} &= \frac{d\boldsymbol{\beta}_{-j}}{d\beta_{j}} \frac{d\beta_{j}}{dt},

where :math:`k = \Phi(1-\alpha)` is the quantile we want to obtain under a normal distribution, and

.. math::

    g = J_{\beta_{j}}^{\top} I^{\boldsymbol{\beta}} J_{\beta_{j}}, \quad J_{\beta_{j}} = \left( \begin{array}{c} 1 \\ \frac{d\boldsymbol{\beta}_{-j}}{d\beta_{j}} \end{array} \right).

Here, :math:`J_{\beta_{j}}` is the Jacobian between :math:`\beta_{j}` and :math:`\boldsymbol{\beta}_{-j}` with the term

.. math::

    \frac{d\boldsymbol{\beta}_{-j}}{d\beta_{j}} = -\left( \frac{\partial^{2} \mathcal{L}}{\partial \boldsymbol{\beta}_{-j}\partial \boldsymbol{\beta}_{-j}^{\top} } \right)^{-1} \frac{\partial^{2} \mathcal{L}}{\partial \beta_{j} \partial \beta_{-j}^{\top}}

and hence the first element is :math:`1` (identity transformation).  :math:`I^{\boldsymbol{\beta}}` is the Fisher information of :math:`\boldsymbol{\beta}`, which is

.. math::

    I^{\boldsymbol{\beta}} = \frac{\partial \boldsymbol{\theta}}{\partial \boldsymbol{\beta}^{\top}} \Sigma^{\boldsymbol{\theta}(\boldsymbol{\beta})} \frac{\partial \boldsymbol{\theta}}{\partial \boldsymbol{\beta}}.

It is simply :math:`\Sigma^{\boldsymbol{\beta}}` if :math:`\boldsymbol{\theta} = \boldsymbol{\beta}`.  Different Fisher information can be used for :math:`\Sigma^{\boldsymbol{\beta}}` such as the expected or observed, at :math:`\hat{\boldsymbol{\beta}}` or :math:`\boldsymbol{\beta}`.  After some trivial algebraic manipulation, we can show that our ode boils downs to

.. math::

    \left[ \begin{array}{c} \frac{d\beta_{j}}{dt} \\ \frac{d\boldsymbol{\beta_{-j}}}{dt} \end{array} \right] = k \left[ \begin{array}{c} 1 \\ -A^{-1}w \end{array} \right] \left( v - w^{\top}A^{-1}w \right)^{-1/2}

where the symbols on the RHS above correspond to partitions in the Fisher information

.. math::

    I^{\boldsymbol{\beta}} = \left[ \begin{array}{cc} v & w^{\top} \\ w & A \end{array} \right].

The integration is perform from :math:`t = 0` to :math:`1` and is all handled internally via :func:`geometric`

.. ipython::

    In [1]: xLGeometric, xUGeometric, xLList, xUList = ci.geometric(objSIR, alpha, xhat, full_output=True)

    In [2]: print(xLGeometric)

    In [3]: print(xUGeometric)

Bootstrap
=========

This is perhaps the favorite method to estimate confidence interval for a lot of people.  Although there are many ways to implement bootstrap, semi-parametric is the only logical choice (even though the underlying assumptions may be violated at times).  As we have only implemented OLS type loss functions in this package, the parametric approach seem to be inappropriate when there is no self--efficiency guarantee.  Non-parametric approach requires at least a conditional independence assumption, something easily violated by our **ode**.  Block bootstrap is an option but we are also aware that the errors of an **ode** can be rather rigid, and consistently over/under estimate at certain periods of time.

When we say semi-parametric, we mean the exchange of errors between the observations.  Let our raw error be

.. math::

    \varepsilon_{i} = y_{i} - \hat{y}_{i}

where :math:`\hat{y}_{i}` will be the prediction under :math:`\hat{\boldsymbol{\theta}}` under our model.  Then we construct a new set of observations via

.. math::

    y_{i}^{\ast} = \hat{y}_{i} + \varepsilon^{\ast}, \quad \varepsilon^{\ast} \sim \mathcal{F}

with :math:`\mathcal{F}` being the empirical distribution of the raw errors.  A new set of parameters :math:`\theta^{\ast}` are then found for the bootstrapped samples, and we obtain the :math:`\alpha` confidence interval by taking the :math:`\alpha/2` quantiles.  Invoke the correspond python function yields our bootstrap estimates. Unlike :func:`asymptotic`, the bounds here are used when estimating the parameters of each bootstrap samples.  An error may be returned if estimation failed for any of the bootstrap samples.

.. ipython::

    In [1]: xLBootstrap, xUBootstrap, setX = ci.bootstrap(objSIR, alpha, xhat, iteration=10, lb=boxBoundsArray[:,0], ub=boxBoundsArray[:,1], full_output=True)

    In [2]: print(xLBootstrap)

    In [3]: print(xUBootstrap)

The additional information here can be used to compute the bias, tail effects and test against the normality assumption.  If desired, a simultaneous confidence interval can also be approximated empirically.  Note however that because we are using a semi--parameter method here, if the model specification is wrong then the resulting estimates for the bias is also wrong.  The confidence interval still has the normal approximation guarantee if number of sample is large.

In this case, because the error in the observation is extremely small, the confidence interval is narrow.

.. ipython::

    In [1]: import pylab as P

    In [2]: f = plt.figure()

    In [3]: n, bins, patches = P.hist(setX[:,0], 50)

    In [4]: P.xlabel(r'Estimates of $\beta$');

    In [5]: P.ylabel('Frequency');

    In [6]: P.title('Estimates under a semi-parametric bootstrap scheme');

    @savefig bootstrapCIHist.png
    In [7]: P.show()

    In [8]: P.close()

Comparison Between Methods
==========================

Although we have shown the numerical values for the confidence interval obtained using different method, it may be hard to comprehend how they vary.  As they say, a picture says a million word, and given that this particular model only has two parameters, we can obtain inspect and compare the methods visually via a contour plot.  The code to perform this is shown below but the code block will not be run to save time and space.

.. ipython ::
    :verbatim:

    In [1]: niter = 1000

    In [2]: randNum = numpy.random.rand(niter,2)*2.0

    In [3]: target = [objSIR.cost(randNum[i,:]) for i in range(niter)]

    In [4]: z = numpy.array(target)

    In [5]: x = randNum[:,0]

    In [6]: y = randNum[:,1]

    In [7]: from scipy.interpolate import griddata

    In [8]: xi = numpy.linspace(0.0, 2.0, 100)

    In [9]: yi = numpy.linspace(0.0, 2.0, 100)

    In [10]: zi = griddata((x, y), numpy.log(z), (xi[None,:], yi[:,None]), method='cubic')

    In [11]: fig = plt.figure()

    In [12]: CS = plt.contour(xi, yi, zi, linewidth=0.5)

    In [13]: plt.clabel(CS, fontsize=10, inline=1);

    In [14]: l0 = plt.scatter(xhat[0], xhat[1], marker='o', c='k', s=30)

    In [15]: l1 = plt.scatter(numpy.append(xL[0], xU[0]), numpy.append(xL[1], xU[1]), marker='x', c='m', s=30)

    In [16]: l2 = plt.scatter(numpy.append(xLBootstrap[0], xUBootstrap[0]), numpy.append(xLBootstrap[1], xUBootstrap[1]), marker='x', c='g', s=30)

    In [17]: l3 = plt.scatter(numpy.append(xLGeometric[0], xUGeometric[0]), numpy.append(xLGeometric[1], xUGeometric[1]), marker='x', c='r', s=30)

    In [19]: plt.legend((l0, l1, l2, l3), ('MLE', 'Asymptotic', 'Bootstrap', 'Geometric'), loc='upper left');

    In [20]: plt.ylabel(r'Estimates of $\gamma$');

    In [21]: plt.xlabel(r'Estimates of $\beta$');

    In [22]: plt.title('Location of the confidence intervals on the likelihood surface');

    In [23]: plt.tight_layout();

    In [24]: plt.show()

    In [25]: plt.close()

In the plot above, the bootstrap confidence interval were so close to the MLE, it is impossible to distinguish the two on such a coarse scale.

Furthermore, because the geometric confidence interval is the result of an integration, we can trace the path that lead to the final output that was shown previously.  Again, we are space conscious (and time constrained) so the code block below will not be run.

.. ipython::
    :verbatim:

    In [1]: fig = plt.figure()

    In [2]: CS = plt.contour(xi, yi, zi, linewidth=0.5)

    In [3]: plt.clabel(CS, fontsize=10, inline=1)

    In [4]: l1 = plt.scatter(xLList[0][:,0], xLList[0][:,1], marker='o', c='m', s=10);

    In [5]: l2 = plt.scatter(xUList[0][:,0], xUList[0][:,1], marker='x', c='m', s=10);

    In [6]: plt.legend((l1, l2), ('Lower CI path', 'Upper CI path'), loc='upper left');

    In [7]: plt.ylabel(r'Estimates of $\gamma$');

    In [8]: plt.xlabel(r'Estimates of $\beta$');

    In [9]: plt.title('Integration path of the geometric confidence intervals on the likelihood surface');

    In [10]: plt.tight_layout();

    In [11]: plt.show()

    In [12]: plt.close()


Profile Likelihood Surface
==========================

To investigate why it was hard to find the profile likelihood confidence interval, we can simply look at the surface (which is simply a line as we are profiling).  We find solution of :eq:`nuisanceOptim` for each :math:`\boldsymbol{\theta}_{-j}` at various points of :math:`\boldsymbol{\theta}`.  Equivalently, we can minimize the original loss function as defined previously, and this is the approach below.  We focus out attention to the parameter :math:`\beta` of our SIR model. The results are not shown here but the existence of a solution to :eq:`obj` is evident by simply *eyeballing* the plots.

.. ipython::
    :verbatim:

    In [1]: numIter = 100

    In [2]: x2 = numpy.linspace(0.0, 2.0, numIter)

    In [3]: funcOut = numpy.linspace(0.0, 2.0, numIter)

    In [4]: ode.parameters = [('beta',0.5), ('gamma',1.0/3.0)]

    In [5]: for i in range(numIter):
       ...:     paramEval = [('beta',x2[i]), ('gamma',x2[i])]
       ...:     ode2 = copy.deepcopy(ode)
       ...:     ode2.parameters = paramEval
       ...:     ode2.initial_values = (x0, t[0])
       ...:     objSIR2 = NormalLoss(x2[i], ode2, x0, t[0], t[1::], yObv.copy(), targetState, target_param='gamma')
       ...:     res = scipy.optimize.minimize(fun=objSIR2.cost,
       ...:                                   jac=objSIR2.gradient,
       ...:                                   x0=x2[i],
       ...:                                   bounds=[(0,2)],
       ...:                                   method='L-BFGS-B')
       ...:     funcOut[i] = res['fun']

    In [10]: fig = plt.figure()

    In [10]: plt.plot(x2, objSIR.cost(xhat) - funcOut)

    In [11]: l1 = plt.axhline(-0.5*qchisq(1 - alpha, df=1), 0, 2, color='r')

    In [12]: plt.ylabel(r'$\mathcal{L}(\hat{\theta}) - \mathcal{L}(\theta \mid \beta)$');

    In [13]: plt.xlabel(r'Fixed value of $\beta$');

    In [14]: plt.title('Difference in objective function between MLE\n and the maximization of the nuisance parameters given the\n parameter of interest, beta in this case');

    In [15]: plt.tight_layout();

    In [16]: plt.legend((l1,), (r'$-0.5\mathcal{X}_{1 - \alpha}^{2}(1)$',), loc='lower right');

    @savefig profileLLMaximizerGivenBeta.png
    In [17]: plt.show() #    @savefig profileLLMaximizerGivenBeta.png

    In [18]: plt.close()

Both the upper and lower confidence interval can be found in the profiling procedure, but the part between of :math:`\beta \in \left[0,\hat{\beta}\right]` is not convex, with :math:`\hat{\beta}` being the MLE.  This non--quadratic profile likelihood is due to the non-identifiability of the model given data [Raue2009]_.  For this particular case, we can fix it simply by introducing additional observations in the form of the :math:`I` state.  We encourage the users to try it out for themselves to confirm.

.. ipython::
    :verbatim:

    In [1]: targetState = ['I', 'R']

    In [2]: targetStateIndex = numpy.array(ode.get_state_index(targetState))

    In [3]: y = solution[1::,targetStateIndex] + numpy.random.normal(0, 0.01, (len(solution[1::,targetStateIndex]), 1))

    In [4]: objSIR = NormalLoss(theta, ode, x0, t[0], t[1::], y.copy(), targetState)

    In [5]: xhat = objSIR.fit(theta, lb=boxBoundsArray[:,0], ub=boxBoundsArray[:,1])

    In [6]: for i in range(numIter):
       ...:     paramEval = [('beta', x2[i]), ('gamma', x2[i])]
       ...:     ode2 = copy.deepcopy(ode)
       ...:     ode2.parameters = paramEval
       ...:     ode2.initial_values = (x0, t[0])
       ...:     objSIR2 = NormalLoss(x2[i], ode2, x0, t[0], t[1::], y.copy(), targetState, target_param='gamma')
       ...:     res = scipy.optimize.minimize(fun=objSIR2.cost,
       ...:                                   jac=objSIR2.gradient,
       ...:                                   x0=x2[i],
       ...:                                   bounds=[(0,2)],
       ...:                                   method='L-BFGS-B')
       ...:     funcOut[i] = res['fun']

    In [10]: fig = plt.figure()

    In [10]: plt.plot(x2, objSIR.cost(xhat) - funcOut);

    In [11]: l1 = plt.axhline(-0.5*qchisq(1 - alpha, df=1), 0, 2, color='r')

    In [12]: plt.ylabel(r'$\mathcal{L}(\hat{\theta}) - \mathcal{L}(\theta \mid \beta)$');

    In [13]: plt.xlabel(r'Fixed value of $\beta$');

    In [14]: plt.title('Profile likelihood curve for the parameter of\n interest with more observation');

    In [15]: plt.tight_layout();

    In [16]: plt.legend((l1,), (r'$-0.5\mathcal{X}_{1 - \alpha}^{2}(1)$',), loc='lower right');

    @savefig profileLLMaximizerGivenBetaMoreObs.png
    In [17]: plt.show() #     @savefig profileLLMaximizerGivenBetaMoreObs.png

    In [18]: plt.close()
