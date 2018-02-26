:func:`.SEIR_Birth_Death`
=========================

Extending it to also include birth death process with equal rate :math:`\mu`

.. math::

    \frac{dS}{dt} &= \mu - \beta SI - \mu S \\
    \frac{dE}{dt} &= \beta SI - (\mu + \alpha) E \\
    \frac{dI}{dt} &= \alpha E - (\mu + \gamma) I \\
    \frac{dR}{dt} &= \gamma I

Same parameters value taken from [Aron1984]_ as the SEIR example above is used here.  Observe how the introduction of a birth and a death process changes the graph even though the rest of the parameters remains the same.

.. ipython::

    In [1]: from pygom import common_models

    In [1]: import matplotlib.pyplot as plt

    In [1]: import numpy

    In [1]: ode = common_models.SEIR_Birth_Death({'beta':1800, 'gamma':100, 'alpha':35.84, 'mu':0.02})

    In [1]: t = numpy.linspace(0, 50, 1001)

    In [1]: x0 = [0.0658, 0.0007, 0.0002, 0.0]

    In [1]: ode.initial_values = (x0, t[0])

    In [1]: solution = ode.integrate(t[1::], full_output=True)

    @savefig common_models_seir_bd.png    
    In [1]: ode.plot()

    In [1]: plt.close()