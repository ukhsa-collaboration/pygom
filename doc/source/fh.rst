.. _fh:

******************
Example: Fitz Hugh
******************

Defining the model
==================

We are going to investigate another classic model here, the FitzHugh-Nagumo, or simply FitzHugh here.  The model has already been defined in :mod:`common_models` so we can load it easily

.. ipython::

    In [1]: from pygom import SquareLoss, common_models
    
    In [2]: import numpy
    
    In [3]: import scipy.integrate, scipy.optimize

    In [4]: import math,time,copy

    In [5]: import matplotlib.pyplot as plt

    In [1]: x0 = [-1.0, 1.0]

    In [2]: t0 = 0

    In [3]: # params

    In [4]: paramEval = [('a',0.2), ('b',0.2), ('c',3.0)]

    In [5]: ode = common_models.FitzHugh(paramEval)

    In [5]: ode.initial_values = (x0, t0)

Define a set of time points and lets see how the two states :math:`V` and :math:`R` are suppose to behave.

.. ipython:: 

    In [6]: t = numpy.linspace(1, 20, 30).astype('float64')

    In [7]: solution = ode.integrate(t)

    @savefig fh_plot.png
    In [8]: ode.plot()

Estimate the parameters
=======================

Obtaining the correct parameters for the FitzHugh model is well known to be difficult, this is because the surface is multimodal.  Although this has been shown many times in the literature, so we will omit the details.  Regardless, we give it a go with some initial guess.  with some luck, we will be able to recover the original parameters.  First, we try it out with only one target state

.. ipython::

    In [26]: theta = [0.5, 0.5, 0.5]

    In [27]: objFH = SquareLoss(theta, ode, x0, t0, t, solution[1::,1], 'R')

    In [28]: boxBounds = [
       ....:              (0.0,5.0),
       ....:              (0.0,5.0),
       ....:              (0.0,5.0)
       ....:             ]

    In [29]: res = scipy.optimize.minimize(fun=objFH.cost,
       ....:                               jac=objFH.sensitivity,
       ....:                               x0=theta,
       ....:                               bounds=boxBounds,
       ....:                               method='L-BFGS-B')

    In [30]: print(res)
       
Then we try the same again but with both state as our target.  Now we won't look at the iterations because they are pretty pointless.

.. ipython::

    In [30]: objFH = SquareLoss(theta, ode, x0, t0, t, solution[1::,:], ['V','R'])

    In [31]: res = scipy.optimize.minimize(fun=objFH.cost,
       ....:                               jac=objFH.sensitivity,
       ....:                               x0=theta,
       ....:                               bounds=boxBounds,
       ....:                               method='L-BFGS-B')

    In [32]: print(res)

Note how the estimates are the same, unlike other models.  

Estimate initial value
======================

We can further assume that we have no idea about the initial values for :math:`V` and :math:`R` as well.  We also provide guesstimate to set off the optimization.  The input vector :math:`\theta` must have the parameters first, then the initial values, along with the corresponding bounds.

First, only a single target state, i.e. we only have observations for one of states which is :math:`R` in this case

.. ipython::

    In [35]: objFH = SquareLoss(theta, ode, x0, t0, t, solution[1::,1], 'R')

    In [35]: boxBounds = [
       ....:              (0.0,5.0),
       ....:              (0.0,5.0),
       ....:              (0.0,5.0),
       ....:              (None,None),
       ....:              (None,None)
       ....:             ]

    In [36]: res = scipy.optimize.minimize(fun=objFH.costIV,
       ....:                               jac=objFH.sensitivityIV,
       ....:                               x0=theta + [-0.5,0.5],
       ....:                               bounds=boxBounds,
       ....:                               method='L-BFGS-B')

    In [37]: print(res)

then both state as target at the same time

.. ipython::

    In [38]: objFH = SquareLoss(theta, ode, x0, t0, t, solution[1::,:], ['V','R'])

    In [38]: res = scipy.optimize.minimize(fun=objFH.costIV,
       ....:                               jac=objFH.sensitivityIV,
       ....:                               x0=theta + [-0.5, 0.5],
       ....:                               bounds=boxBounds,
       ....:                               method='L-BFGS-B')

    In [39]: print(res)

See the difference between the two estimate with the latter, both state were used, yielding superior estimates.  Note that only the forward sensitivity method is implemented when estimating the initial value, and it is assumed that the starting condition for all the states are unknown.  

The choice of algorithm here is the **L-BFGS-B** which is a better choice because the parameter space of the FitzHugh is rough (i.e. large second derivative) as well as being multimodal.  This means that the Hessian is not guaranteed to be positive definite and approximation using :math:`J^{\top}J` is poor, with :math:`J` being the Jacobian of the objective function.


