.. _gradient:

*************************************
Gradient estimation under square loss
*************************************

Assuming that we have a set of :math:`N` observations :math:`y_{i}` at specific time points :math:`t_{i}`, :math:`i = 1,\ldots,N`, we may wish to test out a set of ode to see whether it fits to the data.  The most natural way to test such *fit* is to minimize the sum of squares between our observations :math:`y` and see whether the resulting solution of the ode and the estimationed parameters makes sense.   

We assume that this estimation process will be tackled through a non-linear optimization point of view.  However, it should be noted that such estimates can also be performed via MCMC or from a global optimization perspective.  A key element in non-linear optimization is the gradient, which is the focus of this page.

Multiple ways of obtaining the gradient have been implemented.  All of them serve a certain purpose and may not be a viable/appropriate options depending on the type of ode.  More generally, let :math:`d,p` be the number of states and paramters respectively.  Then finite difference methods have a run order of :math:`O(p+1)` of the original ode, forward sensitivity require an integration of an ode of size :math:`(d+1)p` rather than :math:`d`.  The adjoint method require two run of size :math:`d` in principle, but actual run time is dependent on the number of observations.  

For the details of the classes and methods, please refer to :ref:`mod`.

Notation
========

We introduce the notations that will be used in the rest of the page, some of which may be slightly unconventional but necessary due to the complexity of the problem.  Let :math:`x \in \mathbb{R}^{d}` and :math:`\theta \in \mathbb{R}^{p}` be the states and parameters respectively.  The term *state* or *simulation* are used interchangeably, even though strictly speaking a state is :math:`x`  whereas :math:`x(t)` is the simulation.  An ode is defined as 

.. math::

    f(x,\theta) = \dot{x} = \frac{\partial x}{\partial t}

and usually comes with a set of initial conditions :math:`(x_0,t_0)` where :math:`t_0 \le t_{i} \forall i`.  Let :math:`g(x,\theta)` be a function that maps the set of states to the observations, :math:`g : \mathbb{R}^{d} \rightarrow \mathbb{R}^{m}`.  For compartmental problems, which is our focus, :math:`\nabla_{\theta}g(x,\theta)` is usually zero and :math:`\nabla_{x}g(x,\theta)` is an identity function for some or all of the states :math:`x`.  Denote :math:`l(x_{0},\theta,x)` as our cost function :math:`l : \mathbb{R}^{m} \rightarrow \mathbb{R}` and :math:`L(x_{0},\theta,x)` be the sum of :math:`l(\cdot)`.  Both :math:`x` and :math:`x_{0}` are usually dropped for simplicity.  We will be dealing exclusively with square loss here, which means that 

.. math::

    L(\theta) = \sum_{i=1}^{N} \left\| y_{i} - g(x(t_{i})) \right\|^{2} = \mathbf{e}^{\top} \mathbf{e}

where :math:`\mathbf{e}` is the residual vector, with elements

.. math:: 

    e_{i} = y_{i} - x(t_{i}).


Model setup
===========

Again, we demonstrate the functionalities of our classes using an SIR model.  

.. ipython::

    In [1]: from pygom import SquareLoss, common_models

    In [2]: import copy,time,numpy
    
    In [2]: ode = common_models.SIR()
    
    In [3]: paramEval = [('beta',0.5), ('gamma',1.0/3.0) ]
    
    In [7]: # the initial state, normalized to zero one
    
    In [8]: x0 = [1., 1.27e-6, 0.]

    In [5]: # initial time

    In [6]: t0 = 0

    In [5]: ode.parameters = paramEval

    In [6]: ode.initial_values = (x0, t0)

    In [9]: # set the time sequence that we would like to observe
     
    In [10]: t = numpy.linspace(1, 150, 100)
    
    In [11]: numStep = len(t)

    In [11]: solution = ode.integrate(t)

    In [12]: y = solution[1::,2].copy()

    In [13]: y += numpy.random.normal(0, 0.1, y.shape)

Now we have set up the model along with some observations, obtaining the gradient only requires the end user to put the appropriate information it into the class :class:`SquareLoss`.  Given the initial guess :math:`\theta`

.. ipython::
 
    In [210]: theta = [0.2, 0.2]

We initialize the :class:`SquareLoss` simply as

.. ipython::

    In [20]: objSIR = SquareLoss(theta, ode, x0, t0, t, y, 'R')

where the we also have to specify the state our observations are from.  Now, we demonstrate the different methods in obtaining the gradient and mathematics behind it.

Forward sensitivity
===================

The forward sensitivity equations are derived by differentiating the states implicitly, which yields

.. math::

    \frac{d\dot{x}}{d\theta} = \frac{\partial f}{\partial x}\frac{dx}{d\theta} + \frac{\partial f}{\partial \theta}.

So finding the sensitivies :math:`\frac{dx}{d\theta}` simply require another integration of a :math:`p` coupled ode of :math:`d` dimension, each with the same Jacobian as the original ode.  This integration is performed along with the original ode because of possible non-linearity.

A direct call to the method :meth:`sensitivity <pygom.SquareLoss.sensitivity>` computed the gradient 

.. ipython::
    
    In [33]: gradSens = objSIR.sensitivity()

whereas :meth:`.jac` will allow the end user to obtain the Jacobian (of the objective function) and the residuals, the information required to get the gradient as we see next.

.. ipython:: 

    In [33]: objJac, output = objSIR.jac(full_output=True)


Gradient
========

Just the sensitivities alone are not enough to obtain the gradient, but we are :math:`90\%` there.  Differentiating the loss function 

.. math::

    \frac{dL}{d\theta} &= \nabla_{\theta} \sum_{i=1}^{N}\frac{dl}{dg} \\
                       &= \sum_{i=1}^{N} \frac{\partial l}{\partial x}\frac{dx}{d\theta} + \frac{\partial l}{\partial \theta} \\
                       &= \sum_{i=1}^{N} \frac{\partial l}{\partial g}\frac{\partial g}{\partial x}\frac{dx}{d\theta} + \frac{\partial l}{\partial g}\frac{\partial g}{\partial \theta}

via chain rule.  When :math:`\frac{\partial g}{\partial \theta} = 0`, the total gradient simplifies to 

.. math::

    \frac{dL}{d\theta} = \sum_{i=1}^{N} \frac{\partial l}{\partial g}\frac{\partial g}{\partial x}\frac{dx}{d\theta} 

Obviously, the time indicies are dropped above but all the terms above are evaluated only at the observed time points.  More concretely, this means that 

.. math::

    \frac{\partial l(x(j),\theta)}{\partial g} = \left\{ \begin{array}{ll} -2(y_{i} - x(j)) & , \; j = t_{i} \\ 0 & \; \text{otherwise} \end{array} \right.

When :math:`g(\cdot)` is an identity function (which is assumed to be the case in :class:`SquareLoss`)

.. math::

    \frac{\partial g(x(t_{i}),\theta)}{\partial x} = I_{d}
                       
then the gradient simplifies even further as it is simply

.. math::

    \frac{dL}{d\theta} = -2\mathbf{e}^{\top}\mathbf{S}

where :math:`\mathbf{e}` is the vector of residuals and :math:`\mathbf{S} = \left[\mathbf{s}_{1},\mathbf{s}_{2},\ldots,\mathbf{s}_{n}\right]` with elements

.. math::

    \mathbf{s}_{i} = \frac{dx}{d\theta}(t_{i}),

the solution of the forward sensitivies at time :math:`t_{i}`, obtained from solving the coupled ode as mentioned previously.  

Jacobian
========

Now note how the gradient simplifies to :math:`-2\mathbf{e}^{\top}\mathbf{S}`.  Recall that a standard result in non-linear programming states that the gradient of a sum of sqaures objective function :math:`L(\theta,y,x)` is

.. math:: 

    \nabla_{\theta} L(\theta,y,x) = -2(\mathbf{J}^{T} \left[\mathbf{y} - \mathbf{f}(x,\boldsymbol{\theta}) \right] )^{\top}

with :math:`f(x,\theta)` our non-linear function and :math:`J` our Jacobian with elements

.. math::

    J_{i} = \frac{\partial f(x_{i},\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}.

This is exactly what we have seen previously, substituting in reveals that :math:`J = \mathbf{S}`.  Hence, the Jacobian is (a necessary)by product when we wish to obtain the gradient.  In fact, this is exactly how we proceed in :func:`sensitivity <pygom.SquareLoss.sensitivity>` where it makes an internal call to :func:`jac <pygom.SqaureLoss.jac>` to obtain the Jacobian first.  This allows the end user to have more options when choosing which type of algorithms to use, i.e. Gauss-Newton or Levenberg-Marquardt.

To check that the output is in fact the same

.. ipython::

    In [1]: objJac.transpose().dot(-2*output['resid']) - gradSens

Adjoint
=======

When the number of parameters increases, the number of sensitivies also increases.  The time required scales directly with the number of parameters.  We describe another method which does not depend on the number of parameters, but rather, the number of states and observations.

The full derivations will not be shown here, but we aim to provide enough information to work out the steps performed in the our code.  Let write our optimization problem as

.. math::

    min_{\theta} \quad & \int_{t_{0}}^{T} l(x_{0},\theta,x(t)) dt \\
    s.t. \quad & \dot{x} = f(x,\theta) 

which is identical to the original problem but in a continuous setting.  Now write the constrained problem in the Lagrangian form

.. math::

    min_{\theta} \; L(\theta) + \int_{t_{0}}^{T} \lambda^{\top}(\dot{x} - f(x,\theta))

with Lagrangian multiplier :math:`\lambda \ge 0`.  After some algebraic manipulation, it can be shown that the total derivative of the Lagrangian function is

.. math::

    \frac{dL}{d\theta} = \int_{t_{0}}^{T} \left(\frac{\partial l}{\partial \theta} - \lambda^{\top}\frac{\partial f}{\partial \theta} \right) dt.

Using previously defined loss functions (the identity), the first term is zero and evaluating :math:`\frac{\partial f}{\partial \theta}` is trivial.  What remains is the calculation of :math:`\lambda(t)` for :math:`t \in \left[t_{0},T\right]`.

Although this still seem to be ill-posed problem when Looking at the Lagrangian function, one can actually obtain the *adjoint equation*, after certain assumptions and 

.. math::

    \frac{d\lambda^{\top}}{dt} = \frac{\partial l}{\partial x} - \lambda^{\top}\frac{\partial f}{\partial \theta}.

which is again an integration.  An unfortunate situation arise here for non-linear systems because we use the minus Jacobian in the adjoint equation.  So if the eigenvalues of the Jacobian indicate that our original ode is stable, such as -1, the minus eigenvalues (now 1) implies that the adjoint equation is not stable.  Therefore, one must integrate backward in time to solve the adjoint equation and it cannot be solved simultaneously as the ode, unlike the forward sensitivity equations.  

Given a non-linearity ode, we must store information about the states between :math:`t_{0}` and :math:`T` in order to perform the integration.  There are two options, both require storing many evaluated :math:`x(j)` within the interval :math:`\left[t_{0},T\right]`.  Unfortunately, only one is available; interpolation over all states and integrate using the interpolating functions.  The alternative of using observed :math:`x(j)'s` at fixed points is not competitive because we are unable to use fortran routines for the integration

The method of choice here to perform the adjoint calcuation is to run a forward integration, then perform an interpolation using splines with explicit knots at the observed time points. 

.. ipython::

    In [326]: odeSIRAdjoint, outputAdjoint = objSIR.adjoint(full_output=True)

This is because evaluating the Jacobian may be expensive and Runge-kutta method suffers as the complexity increases.  In non-linear model such as those found in epidemiology, each element of the Jacobian may be the result of a complicated equation where linear step method will shine as it makes as little function evaluation as possible.  
Note that derivations in the literature, the initial condition when evaluating the adjoint equation is :math:`\lambda(T)=0`.  But in our code we used :math:`\lambda(T) = -2(y(T)-x(T))`. Recall that we have observation :math:`y(T)` and simulation :math:`x(T)`, so that the adjoint equation evaluated at time :math:`T`

.. math::

    \frac{\partial \lambda^{\top}}{\partial t} \Big|_{T} = -2(y-f(x,\theta))\Big|_{T}  - \lambda(T)\frac{\partial f}{\partial \theta}\Big|_{T}

with the second term equal to zero.  Integration under step size :math:`h` implies that :math:`\lambda(T) \approx \lim_{h \to 0} \lambda(T-h) = -2(y(T)-x(T))`.

Time Comparison
===============

A simple time comparison between the different methods reveals that the forward sensitivity method dominates the others by a wide margin.  It will be tempting to conclude that it is the best and should be the default at all times but that is not true, due to the complexity of each method mentioned previously.  We leave it to the end user to find out the best method for their specific problem.

.. ipython::

    In [319]: %timeit gradSens = objSIR.sensitivity()

    In [326]: %timeit odeSIRAdjoint,outputAdjoint = objSIR.adjoint(full_output=True)


Hessian
=======

The Hessian is defined by

.. math::

    \frac{\partial^{2} l}{\partial \theta^{2}} = \left( \frac{\partial l}{\partial x} \otimes I_{p} \right) \frac{\partial^{2} x}{\partial \theta^{2}} + \frac{\partial x}{\partial \theta}^{\top}\frac{\partial^{2} l}{\partial x^{2}}\frac{\partial x}{\partial \theta}

where :math:`\otimes` is the Kronecker product.  Note that :math:`\nabla_{\theta} x` is the sensitivity and the second order sensitivities can be found again via the forward method, which involve another set of ode's, namely the forward-forward sensitivities

.. math::

    \frac{\partial}{\partial t}\left(\frac{\partial^{2} x}{\partial \theta^{2}}\right) = \left( \frac{\partial f}{\partial x} \otimes I_{p} \right) \frac{\partial^{2} x}{\partial \theta^{2}} + \left( I_{d} \otimes \frac{\partial x}{\partial \theta}^{\top} \right) \frac{\partial^{2} f}{\partial x^{2}} \frac{\partial x}{\partial \theta}.

From before, we know that

.. math::

    \frac{\partial l}{\partial x} = (-2y+2x)  \quad and \quad \frac{\partial^{2} l}{\partial x^{2}} = 2I_{d}

so our Hessian reduces to 

.. math::

    \frac{\partial^{2} l}{\partial \theta^{2}} = \left( \left(-2y+2x\right) \otimes I_{p} \right) \frac{\partial^{2} x}{\partial \theta^{2}} + 2S^{\top}S,

where the second term is a good approximation to the Hessian as mentioned previously.  This is the only implementation in place so far even though obtaining the estimate this way is relatively slow.  

Just to demonstate how it works, lets look at the Hessian at the optimal point.  First, we obtain the optimal value

.. ipython:: 

    In [211]: import scipy.linalg,scipy.optimize

    In [212]: boxBounds = [(0.0, 2.0), (0.0, 2.0)]

    In [213]: res = scipy.optimize.minimize(fun=objSIR.cost,
       .....:                               jac=objSIR.sensitivity,
       .....:                               x0=theta,
       .....:                               bounds=boxBounds,
       .....:                               method='L-BFGS-B')

Then compare again the least square estimate of the covariance matrix against our version

.. ipython::

    In [211]: resLS, cov_x, infodict, mesg, ier = scipy.optimize.leastsq(func=objSIR.residual, x0=res['x'], full_output=True)

    In [212]: HJTJ, outputHJTJ = objSIR.hessian(full_output=True)

    In [311]: print(scipy.linalg.inv(HJTJ))

    In [312]: print(cov_x)

also note the difference between the Hessian and the approximation using the Jacobian, which is in fact what the least squares routine uses.

.. ipython::

    In [313]: print(scipy.linalg.inv(outputHJTJ['JTJ']))
