:func:`.Lotka_Volterra`
=======================

A standard Lotka-Volterra (preditor and prey) model with two states and four parameters [Lotka1920]_.

.. math::
        
    \frac{dx}{dt} &= \alpha x - cxy \\
    \frac{dy}{dt} &= -\delta y + \gamma xy

with both birth and death processes.

.. ipython::

    In [1]: from pygom import common_models

    In [1]: import numpy

    In [1]: import matplotlib.pyplot as plt

    In [1]: x0 = [2.0, 6.0]

    In [1]: ode = common_models.Lotka_Volterra({'alpha':1, 'delta':3, 'c':2, 'gamma':6})

    In [1]: ode.initial_values = (x0, 0)

    In [1]: t = numpy.linspace(0.1, 100, 10000)

    In [1]: solution = ode.integrate(t)

    @savefig common_models_Lotka_Volterra.png
    In [1]: ode.plot()

    In [1]: plt.close()

Then we generate the graph at `Wolfram Alpha <http://www.wolframalpha.com/input/?i=lotka-volterra+equations>`_ with varying initial conditions.  

.. ipython::

    In [1]: x1List = numpy.linspace(0.2, 2.0, 5)

    In [1]: x2List = numpy.linspace(0.6, 6.0, 5)

    In [1]: fig = plt.figure()

    In [1]: solutionList = list()

    In [1]: ode = common_models.Lotka_Volterra({'alpha':1, 'delta':3, 'c':2, 'gamma':6})

    In [1]: for i in range(len(x1List)):
       ...:     ode.initial_values = ([x1List[i], x2List[i]], 0)
       ...:     solutionList += [ode.integrate(t)]

    In [1]: for i in range(len(x1List)): plt.plot(solutionList[i][100::,0], solutionList[i][100::,1], 'b')

    In [1]: plt.xlabel('x')

    In [1]: plt.ylabel('y')

    @savefig common_models_Lotka_Volterra_initial_condition.png    
    In [1]: plt.show()

    In [1]: plt.close()

We also know that the system has the critical points at :math:`x = \delta / \gamma` and :math:`y=\alpha / c`. If we changes the parameters in such a way that the ration between :math:`x` and :math:`y` remains the same, then we get a figure as below

.. ipython::

    In [1]: cList = numpy.linspace(0.1, 2.0, 5)

    In [1]: gammaList = numpy.linspace(0.6, 6.0, 5)

    In [1]: fig = plt.figure()

    In [1]: for i in range(len(x1List)):
       ...:     ode = common_models.Lotka_Volterra({'alpha':1, 'delta':3, 'c':cList[i], 'gamma':gammaList[i]})
       ...:     ode.initial_values = (x0, 0)
       ...:     solutionList += [ode.integrate(t)]

    In [1]: for i in range(len(cList)): plt.plot(solutionList[i][100::,0], solutionList[i][100::,1])

    In [1]: plt.xlabel('x')

    In [1]: plt.ylabel('y')

    @savefig common_models_Lotka_Volterra_critical_point.png
    In [1]: plt.show()

    In [1]: plt.close()

where all the cycles goes through the same points.
