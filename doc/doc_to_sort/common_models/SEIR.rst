:func:`.SEIR`
=============

A natural extension to the SIR is the SEIR model. An extra parameter :math:`\alpha`, which is the inverse of the incubation period is introduced.

.. math::
    
    \frac{dS}{dt} &= -\beta SI \\

    \frac{dE}{dt} &= \beta SI - \alpha E \\

    \frac{dI}{dt} &= \alpha E - \gamma I \\

    \frac{dR}{dt} &= \gamma I 
    
We use the parameters from [Aron1984] here to generate our plots, which does not yield a *nice* and *sensible* epidemic curve as the birth and death processes are missing.
    
.. ipython::

    In [1]: from pygom import common_models

    In [1]: import numpy

    In [1]: ode = common_models.SEIR({'beta':1800, 'gamma':100, 'alpha':35.84})
    
    In [1]: t = numpy.linspace(0, 50, 1001)
    
    In [1]: x0 = [0.0658, 0.0007, 0.0002, 0.0]

    In [1]: ode.initial_values = (x0, t[0])

    In [1]: solution = ode.integrate(t[1::])
    
    @savefig common_models_seir.png
    In [1]: ode.plot()

