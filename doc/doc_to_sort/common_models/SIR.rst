:func:`.SIR`
============

A standard SIR model defined by the equations
    
.. math::
    
    \frac{dS}{dt} &= -\beta SI \\
    \frac{dI}{dt} &= \beta SI - \gamma I \\
    \frac{dR}{dt} &= \gamma I
    
Note that the examples and parameters are taken from [Brauer2008]_, namely Figure 1.4.  Hence, the first one below may not appear to make much sense.
    
.. ipython::
    
    In [1]: from pygom import common_models

    In [1]: import numpy

    In [1]: ode = common_models.SIR({'beta':3.6, 'gamma':0.2})
    
    In [1]: t = numpy.linspace(0, 730, 1001)
    
    In [1]: N = 7781984.0
    
    In [1]: x0 = [1.0, 10.0/N, 0.0]

    In [1]: ode.initial_values = (x0, t[0])

    In [1]: solution = ode.integrate(t[1::])
    
    @savefig common_models_sir.png  
    In [1]: ode.plot()

Now we have the more sensible plot, where the initial susceptibles is only a fraction of 1.

.. ipython::

    In [1]: x0 = [0.065, 123*(5.0/30.0)/N, 0.0]

    In [1]: ode.initial_values = (x0, t[0])

    In [1]: solution = ode.integrate(t[1::])
    
    @savefig common_models_sir_realistic.png  
    In [1]: ode.plot()

