:func:`.SEIR`
=============

A natural extension to the SIR is the SEIR model. An extra parameter :math:`\alpha`, which is the inverse of the incubation period is introduced.

.. math::
    
    \frac{dS}{dt} &= -\beta SI \\
    
    \frac{dE}{dt} &= \beta SI - \alpha E \\
    
    \frac{dI}{dt} &= \alpha E - \gamma I \\
    
    \frac{dR}{dt} &= \gamma I 
    
We use the parameters from [2] here to generate our plots.    
    
.. ipython::

    In [1]: from pygom import common_models

    In [1]: import numpy

    In [1]: ode = common_models.SEIR({'beta':1800,'gamma':100,'alpha':35.84})
    
    In [1]: t = numpy.linspace(0,50,1001)
    
    In [1]: x0 = [0.0658,0.0007,0.0002,0.0]
    
    In [1]: solution = ode.setInitialValue(x0,t[0]).integrate(t[1::])
    
    @savefig common_models_seir.png
    In [1]: ode.plot()

**References**

[1] Seasonality and period-doubling bifurcations in an epidemic model, Aron J.L. and Schwartz I.B., Journal of Theoretical Biology, Volume 110, Issue 4, pg 665-679, 1984
