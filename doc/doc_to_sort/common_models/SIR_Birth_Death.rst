:func:`.SIR_Birth_Death`
========================

Next, we look at an SIR model with birth death

.. math::

    \frac{dS}{dt} &= B -\beta SI - \mu S \\
    \frac{dI}{dt} &= \beta SI - \gamma I - \mu I \\
    \frac{dR}{dt} &= \gamma I
        
        
Continuing from the example above, but now with a much longer time frame.  Note that the birth and death rate are the same.

.. ipython:: 
    
    In [1]: from pygom import common_models

    In [1]: import numpy

    In [1]: B = 126372.0/365.0
    
    In [1]: N = 7781984.0
    
    In [1]: ode = common_models.SIR_Birth_Death({'beta':3.6, 'gamma':0.2, 'B':B/N, 'mu':B/N})
    
    In [1]: t = numpy.linspace(0, 35*365, 10001)
    
    In [1]: x0 = [0.065, 123.0*(5.0/30.0)/N, 0.0]

    In [1]: ode.initial_values = (x0, t[0])
    
    In [1]: solution = ode.integrate(t[1::])
    
    @savefig common_models_sir_bd.png  
    In [1]: ode.plot()
    
