:func:`.SIS_Periodic`
=====================

Now we look at an extension of the SIS model by incorporating periodic contact rate.  Note how our equation is defined by a single ode for state **I**.

.. math::

    \frac{dI}{dt} = (\beta(t)N - \alpha) I - \beta(t)I^{2}
	
where :math:`\beta(t) = 2 - 1.8 \cos(5t)`.  As the name suggests, it achieves a (stable) periodic solution.  Note how the plots have two sub-graphs, where :math:`\tau` is in fact our time component which we have taken out of the original equation when converting it to a automonous system.   

.. ipython:: 

    In [1]: from pygom import common_models

    In [1]: import matplotlib.pyplot as plt

    In [1]: import numpy

    In [1]: ode = common_models.SIS_Periodic({'alpha':1.0})

    In [1]: t = numpy.linspace(0, 10, 101)

    In [1]: x0 = [0.1,0.]

    In [1]: ode.initial_values = (x0, t[0])

    In [1]: solution = ode.integrate(t[1::])

    @savefig common_models_sis_periodic.png 
    In [1]: ode.plot()

    In [1]: plt.close()