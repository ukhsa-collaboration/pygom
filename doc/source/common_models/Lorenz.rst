:func:`.Lorenz`
===============

The Lorenz attractor

.. math::
    
    \frac{dx}{dt} &= \sigma (y-x) \\
    \frac{dy}{dt} &= x (\rho - z) - y \\
    \frac{dz}{dt} &= xy - \beta z
    
A classic example is 

.. ipython::

    In [1]: from pygom import common_models
    
    In [1]: import numpy
    
    In [1]: import matplotlib.pyplot as plt
    
    In [1]: t = numpy.linspace(0,100,20000)
    
    In [1]: ode = common_models.Lorenz({'beta':8.0/3.0,'sigma':10.0,'rho':28.0}).setInitialValue([1.,1.,1.],t[0])
    
    In [1]: solution = ode.integrate(t[1::])
    
    In [1]: plt.plot(solution[:,0],solution[:,2]);

    @savefig common_models_Lorenz.png
    In [1]: plt.show()


**Reference**

[1] Deterministic Nonperiodic Flow, Lorenz, Edward N., Journal of the Atmospheric Sciences, Volume 20, Issus 2, pgs 130-141, 1963
