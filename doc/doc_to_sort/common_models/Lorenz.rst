:func:`.Lorenz`
===============

The Lorenz attractor [Lorenz1963]_ defined by the equations

.. math::
    
    \frac{dx}{dt} &= \sigma (y-x) \\
    \frac{dy}{dt} &= x (\rho - z) - y \\
    \frac{dz}{dt} &= xy - \beta z
    
A classic example is 

.. ipython::

    In [1]: from pygom import common_models
    
    In [1]: import numpy
    
    In [1]: import matplotlib.pyplot as plt
    
    In [1]: t = numpy.linspace(0, 100, 20000)
    
    In [1]: ode = common_models.Lorenz({'beta':8.0/3.0, 'sigma':10.0, 'rho':28.0})

    In [1]: ode.initial_values = ([1., 1., 1.], t[0])

    In [1]: solution = ode.integrate(t[1::])

    In [1]: f = plt.figure()
    
    In [1]: plt.plot(solution[:,0], solution[:,2]);

    @savefig common_models_Lorenz.png
    In [1]: plt.show()


