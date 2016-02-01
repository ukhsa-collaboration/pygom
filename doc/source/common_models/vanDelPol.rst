:func:`.vanDelPol`
==================

The van Del Pol oscillator

.. math::
    
    \frac{dx}{dt} &= \sigma (y-x) \\
    \frac{dy}{dt} &= x (\rho - z) - y \\
    \frac{dz}{dt} &= xy - \beta z
    
A classic example is 

.. ipython::

    In [1]: from pygom import common_models
    
    In [1]: import numpy
    
    In [1]: import matplotlib.pyplot as plt
    
    In [1]: t = numpy.linspace(0,20,1000)
    
    In [1]: ode = common_models.vanDelPol({'mu':1.0}).setInitialValue([2.0,0.0],t[0])
    
    In [1]: solution = ode.integrate(t[1::])
    
    @savefig common_models_vanDelPol.png
    In [1]: ode.plot()
    
    In [1]: plt.close()

    In [1]: plt.plot(solution[:,0],solution[:,1]);

    @savefig common_models_vanDelPol_yprime_y_1.png    
    In [1]: plt.show()
    
    In [1]: plt.close()

When we change the value, as per `Wolfram <http://mathworld.wolfram.com/vanderPolEquation.html>`_  

.. ipython::
	
    In [1]: t = numpy.linspace(0,100,1000)

    In [1]: ode = ode.setParameters({'mu':1.0}).setInitialValue([0.0,0.2],t[0])
	
    In [1]: solution = ode.integrate(t[1::])
    
    In [1]: plt.plot(solution[:,0],solution[:,1]);

    @savefig common_models_vanDelPol_yprime_y_2.png    
    In [1]: plt.show()
    
    In [1]: plt.close()
    
    In [1]: ode = ode.setParameters({'mu':0.2}).setInitialValue([0.0,0.2],t[0])
	
    In [1]: solution = ode.integrate(t[1::])

    In [1]: plt.plot(solution[:,0],solution[:,1]);
    
    @savefig common_models_vanDelPol_yprime_y_3.png
    In [1]: plt.show()
    
    In [1]: plt.close()

**Reference**

[1] On Relaxed Oscillations, van der Pol, Balthasar, The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, Volume 2, Issue 11, pg.  978-992, 1926
