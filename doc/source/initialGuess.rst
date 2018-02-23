.. _initialGuess:

*******************************************
Obtaining good initial value for parameters
*******************************************

Function Interpolation 
======================

When we want to fit the model to data, one of the necessary steps is to supply the optimization procedure a good set of initial guess for the parameters :math:`\theta`.  This may be a challenge when we do have a good understanding of the process we are trying to model i.e. infectious disease may all follow the same SIR process but with vastly different incubation period.

A method to obtain such initial guess based on the collocation is available in this package.  A restriction is that data must be present for all states.   We demonstrate this using the FitzHugh-Nagumo model.


.. ipython::

    In [1]: from pygom import SquareLoss, common_models, get_init
    
    In [2]: import numpy

    In [3]: x0 = [-1.0, 1.0]

    In [4]: t0 = 0

    In [5]: # params

    In [6]: paramEval = [('a',0.2), ('b',0.2), ('c',3.0)]

    In [7]: ode = common_models.FitzHugh(paramEval)

    In [8]: ode.initial_values = (x0, t0)

    In [8]: t = numpy.linspace(1, 20, 30).astype('float64')

    In [9]: solution = ode.integrate(t)

Below, we try to find the initial guess without supplying any further information.  The underlying method fits a cubic spline against the observation and tries to minimize the difference between the first derivative of the spline and the function of the ode.  Varying degree of smoothness penalty is applied to the spline and the best set of parameters is the ones that yields the smallest total error, combining both the fit of the spline against data and the spline against the ode.

.. ipython::

    In [10]: theta, sInfo = get_init(solution[1::,:], t, ode, theta=None, full_output=True)
    
    In [11]: print(theta)
    
    In [12]: print(sInfo)
    
As seen above, we have obtained a very good guess of the parameters, in fact almost the same as the generating process.  The information regarding the smoothing factor shows that the amount of penalty used is small, which is expected given that we use the solution of the ode as observations.  
