:func:`.FitzHugh`
-----------------

The FitzHugh model [1] without external external stimulus.  This is a commonly used model when developing new methodology with regard to ode's, see [2] and [3] and reference therein.

.. math::

    \frac{dV}{dt} &=  c ( V - \frac{V^{3}}{3} + R) \\
    \frac{dR}{dt} &= -\frac{1}{c}(V - a + bR).
    
An example would be 

.. ipython::

    In [1]: import numpy

    In [1]: from pygom import common_models

    In [1]: import matplotlib.pyplot as plt

    In [1]: ode = common_models.FitzHugh({'a':0.2,'b':0.2,'c':3.0})
    
    In [1]: t = numpy.linspace(0,20,101)
    
    In [1]: x0 = [1.0,-1.0]
    
    In [1]: solution = ode.setInitialValue(x0,t[0]).integrate(t[1::])
    
    @savefig common_models_fh_1.png
    In [1]: ode.plot()

    In [1]: plt.close()

    In [1]: fig = plt.figure()

    In [1]: plt.plot(solution[:,0],solution[:,1],'b')

    @savefig common_models_fh_2.png
    In [1]: plt.show()

    In [1]: plt.close()

**References**

[1] Impulses and Physiological States in Theoretical Models of Nerve Membrane, Biophysical Journal, FitzHugh Richard, Volume 1, Issue 6, pg. 445-466, 1961.

[2] Parameter estimation for differential equations: a generalized smoothing approach, Journal of the Royal Statistical Society Series B, Ramsay et al., Volume 69, Issue 5, pg. 741-796, 2007

[3] Riemann manifold Langevin and Hamiltonian Monte Carlo methods, Girolami Mark and  Calderhead Ben, Journal of the Royal Statistical Society Series B, Volume 73, Issue 2, pg. 123-214, 2011.
