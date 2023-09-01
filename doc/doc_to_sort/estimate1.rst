.. _estimate1:

*******************************
Example: Parameter Estimation 1
*******************************

Estimation under square loss
============================

To ease the estimation process when given data, a separate module :mod:`ode_loss` has been constructed for observations coming from a single state.  We demonstrate how to do it via two examples, first, a standard SIR model, then the Legrand SEIHFR model from [Legrand2007]_ used for Ebola in :ref:`estimate2`.

SIR Model
---------

We set up an SIR model as seen previously in :ref:`sir`.

.. ipython::

    In [176]: from pygom import SquareLoss, common_models

    In [179]: import numpy

    In [180]: import scipy.integrate

    In [184]: import matplotlib.pyplot 

    In [185]: # Again, standard SIR model with 2 parameter.  See the first script!

    In [191]: # define the parameters

    In [192]: paramEval = [('beta',0.5), ('gamma',1.0/3.0)]

    In [189]: # initialize the model

    In [190]: ode = common_models.SIR(paramEval)


and we assume that we have perfect information about the :math:`R` compartment.

.. ipython::

    In [196]: x0 = [1, 1.27e-6, 0]

    In [197]: # Time, including the initial time t0 at t=0

    In [198]: t = numpy.linspace(0, 150, 1000)

    In [200]: # Standard.  Find the solution.

    In [201]: solution = scipy.integrate.odeint(ode.ode, x0, t)

    In [202]: y = solution[:,1:3].copy()

Initialize the class with some initial guess

.. ipython::

    In [209]: # our initial guess

    In [210]: theta = [0.2, 0.2]

    In [176]: objSIR = SquareLoss(theta, ode, x0, t[0], t[1::], y[1::,:], ['I','R'])

Note that we need to provide the initial values, :math:`x_{0}` and :math:`t_{0}` differently to the observations :math:`y` and the corresponding time :math:`t`.  Additionally, the state which the observation lies needs to be specified.  Either a single state, or multiple states are allowed, as seen above.

Difference in gradient
----------------------

We have provided two different ways of obtaining the gradient, these are explained in :ref:`gradient` in a bit more detail.  First, lets see how similar the output of the two methods are

.. ipython::

    In [22]: objSIR.sensitivity()
    
    In [25]: objSIR.adjoint()
    
and the time required to obtain the gradient for the SIR model under :math:`\theta = (0.2,0.2)`, previously entered.
   
.. ipython::

    In [22]: %timeit objSIR.sensitivity()

    In [25]: %timeit objSIR.adjoint()

Obviously, the amount of time taken for both method is dependent on the number of observations as well as the number of states.  The effect on the adjoint method as the number of observations differs can be quite evident.  This is because the adjoint method is under a discretization which loops in Python where as the forward sensitivity equations are solved simply via an integration.  As the number of observation gets larger, the affect of the Python loop becomes more obvious. 

Difference in gradient is larger when there are less observations.  This is because the adjoint method use interpolations on the output of the ode between each consecutive time points.  Given solution over the same length of time, fewer discretization naturally leads to a less accurate interpolation.  Note that the interpolation is currently performed using univaraite spline, due to the limitation of python packages.  Ideally, one would prefer to use an (adaptive) Hermite or Chebyshev interpolation.  Note how we ran the two gradient functions once before timing it, that is because we only find the properties (Jacobian, gradient) of the ode during runtime.

Optimized result
----------------

Then standard optimization procedures with some suitable initial guess should yield the correct result.   It is important to set the boundaries for compartmental models as we know that all the parameters are strictly positive.  We put a less restrictive inequality here for demonstration purpose.

.. ipython::

    In [211]: # what we think the bounds are

    In [212]: boxBounds = [(0.0,2.0),(0.0,2.0)]

Then using the optimization routines in :mod:`scipy.optimize`, for example, the *SLSQP* method with the gradient obtained by forward sensitivity.

.. ipython::

    In [208]: from scipy.optimize import minimize

    In [213]: res = minimize(fun=objSIR.cost,
       .....:                jac=objSIR.sensitivity,
       .....:                x0=theta,
       .....:                bounds=boxBounds,
       .....:                method='SLSQP')

    In [214]: print(res)

Other methods available in :func:`scipy.optimize.minimize` can also be used, such as the *L-BFGS-B* and *TNC*.  We can also use methods that accepts the exact Hessian such as *trust-ncg* but that should not be necessary most of the time.
   
