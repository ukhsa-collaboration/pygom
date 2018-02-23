:func:`.vanDelPol`
==================

The van Del Pol oscillator [vanderpol1926]_

.. math::
    
    \frac{dx}{dt} &= \sigma (y-x) \\
    \frac{dy}{dt} &= x (\rho - z) - y \\
    \frac{dz}{dt} &= xy - \beta z
    
A classic example is 

.. ipython::

    In [1]: from pygom import common_models

    In [1]: import numpy

    In [1]: import matplotlib.pyplot as plt

    In [1]: t = numpy.linspace(0, 20, 1000)

    In [1]: ode = common_models.vanDelPol({'mu':1.0})

    In [1]: ode.initial_values = ([2.0, 0.0], t[0])

    In [1]: solution = ode.integrate(t[1::])

    @savefig common_models_vanDelPol.png
    In [1]: ode.plot()

    In [1]: plt.close()

    In [1]: f = plt.figure()

    In [1]: plt.plot(solution[:,0], solution[:,1]);

    @savefig common_models_vanDelPol_yprime_y_1.png    
    In [1]: plt.show()

    In [1]: plt.close()

When we change the value, as per `Wolfram <http://mathworld.wolfram.com/vanderPolEquation.html>`_  

.. ipython::

    In [1]: t = numpy.linspace(0, 100, 1000)

    In [1]: ode.parameters = {'mu':1.0}

    In [1]: ode.initial_values = ([0.0, 0.2], t[0])

    In [1]: solution = ode.integrate(t[1::])

    In [1]: f = plt.figure()

    In [1]: plt.plot(solution[:,0],solution[:,1]);

    @savefig common_models_vanDelPol_yprime_y_2.png    
    In [1]: plt.show()

    In [1]: plt.close()

    In [1]: ode.parameters = {'mu':0.2}

    In [1]: ode.initial_values = ([0.0, 0.2], t[0])

    In [1]: solution = ode.integrate(t[1::])

    In [1]: f = plt.figure()

    In [1]: plt.plot(solution[:,0], solution[:,1]);
    
    @savefig common_models_vanDelPol_yprime_y_3.png
    In [1]: plt.show()
    
    In [1]: plt.close()
