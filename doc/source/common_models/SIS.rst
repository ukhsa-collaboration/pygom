:func:`.SIS`
============

A standard SIS model without the total population :math:`N`. We assume here that :math:`S + I = N` so we can always normalize to 1.  Evidently, the state **S** is not required for understanding the model because it is a deterministic function of state **I**.

.. math::

    \frac{dS}{dt} &=  -\beta S I + \gamma I \\
    \frac{dI}{dt} &= \beta S I - \gamma I.
    
An example would be 

.. ipython::

    In [1]: from pygom import common_models

    In [1]: import matplotlib.pyplot as plt

    In [1]: import numpy

    In [1]: ode = common_models.SIS({'beta':0.5,'gamma':0.2})
    
    In [1]: t = numpy.linspace(0,20,101)
    
    In [1]: x0 = [1.0,0.1]
    
    In [1]: solution = ode.setInitialValue(x0,t[0]).integrate(t[1::])
    
    @savefig common_models_sis.png    
    In [1]: ode.plot()

    In [1]: plt.close()

