:func:`.SEIR_Birth_Death_Periodic`
==================================

Now extending the SEIR to also have periodic contact, as in [Aron1984]_.

.. math::
    
    \frac{dS}{dt} &= \mu - \beta(t)SI - \mu S \\
    \frac{dE}{dt} &= \beta(t)SI - (\mu + \alpha) E \\
    \frac{dI}{dt} &= \alpha E - (\mu + \gamma) I \\
    \frac{dR}{dt} &= \gamma I.

.. ipython:: 
    
    In [1]: from pygom import common_models

    In [1]: import numpy

    In [1]: import matplotlib.pyplot as plt

    In [1]: ode = common_models.SEIR_Birth_Death_Periodic({'beta_0':1800, 'beta_1':0.2, 'gamma':100, 'alpha':35.84, 'mu':0.02})
    
    In [1]: t = numpy.linspace(0, 50, 1001)

    In [1]: ode.initial_values = (x0, t[0])
    
    In [1]: x0 = [0.0658, 0.0007, 0.0002, 0.0]
    
    In [1]: solution = ode.integrate(t[1::])
    
    @savefig common_models_seir_bd_periodic1.png
    In [1]: ode.plot()

    In [1]: plt.close()
	
The periodicity is obvious when looking at the the plot between states :math:`S` and :math:`E`, in logarithmic scale.

.. ipython::

    In [1]: fig = plt.figure();
   	    
    In [1]: plt.plot(numpy.log(solution[:,0]), numpy.log(solution[:,1]));
   
    In [1]: plt.xlabel('log of S');

    In [1]: plt.ylabel('log of E');

    @savefig common_models_seir_bd_periodic2.png
    In [1]: plt.show()
        
    In [1]: plt.close()

Similarly, we can see the same thing between the states :math:`E` and :math:`I`.

.. ipython::

    In [1]: fig = plt.figure();
    
    In [1]: plt.plot(numpy.log(solution[:,1]), numpy.log(solution[:,2]));
    
    In [1]: plt.xlabel('log of E');

    In [1]: plt.ylabel('log of I');

    @savefig common_models_seir_bd_periodic3.png
    In [1]: plt.show()

    In [1]: plt.close()

