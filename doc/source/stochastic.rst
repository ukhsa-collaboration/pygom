.. _stochastic:

********************************
Stochastic representation of ode
********************************

There are multiple interpretation of stochasticity of a deterministic ode.  We have implemented two of the most common interpretation; when the parameters are realizations of some underlying distribution, and when we have a so called chemical master equation where each transition represent a jump.  Again, we use the standard SIR example as previously seen in ref:`sir`.

.. ipython::

    In [1]: from pygom import SimulateOdeModel, Transition, TransitionType

    In [1]: import matplotlib.pyplot as plt

    In [1]: import numpy

    In [1]: x0 = [1, 1.27e-6, 0]

    In [1]: t = numpy.linspace(0, 150, 100)

    In [1]: stateList = ['S', 'I', 'R']

    In [1]: paramList = ['beta', 'gamma']

    In [1]: transitionList = [
       ...:                   Transition(origState='S', destState='I', equation='beta*S*I', transitionType=TransitionType.T),
       ...:                   Transition(origState='I', destState='R', equation='gamma*I', transitionType=TransitionType.T)
       ...:                   ]

    In [1]: odeS = SimulateOdeModel(stateList,paramList,transitionList=transitionList)

    In [1]: odeS = odeS.setParameters([0.5, 1.0/3.0]).setInitialValue(x0, t[0])

    In [1]: solutionReference = odeS.integrate(t[1::], full_output=False)


Stochastic Parameter
====================

In our first scenario, we assume that the parameters follow some underlying distribution.  Given that both :math:`\beta` and :math:`\gamma` in our SIR model has to be non-negative, it seemed natural to use a Gamma distribution.  We make use of the familiar syntax from `R <http://www.r-project.org/>`_ to define our distribution.  Unfortunately, we have to define it via a tuple, where the first is the function handle (name) while the second the parameters.  Note that the parameters can be defined as either a dictionary or as the same sequence as `R <http://www.r-project.org/>`_, which is the shape then the rate in the Gamma case.

.. ipython::

    In [1]: from pygom.utilR import rgamma

    In [1]: d = dict()

    In [1]: d['beta'] = (rgamma,{'shape':100.0, 'rate':200.0})

    In [1]: d['gamma'] = (rgamma,(100.0, 300.0))

    In [1]: odeS.setParameters(d)

    In [1]: Ymean,Yall = odeS.simulateParam(t[1::], 10, full_output=True)

Note that a message is printed above where it is trying to connect to an mpi backend, as our module has the capability to compute in parallel using the IPython.   We have simulated a total of 10 different solutions using different parameters, the plots can be seen below

.. ipython::

    In [1]: f, axarr = plt.subplots(1,3)

    In [1]: for solution in Yall:
       ...:     axarr[0].plot(t, solution[:,0])
       ...:     axarr[1].plot(t, solution[:,1])
       ...:     axarr[2].plot(t, solution[:,2])

    @savefig stochastic_param_all.png
    In [1]: plt.show()

    In [1]: plt.close()

We then see how the expected results, using the sample average of the simulations

.. math::

   \tilde{x}(T) = \mathbb{E}\left[ \int_{t_{0}}^{T} f(\theta,x,t) dt \right]

differs from the reference solution

.. math::

    \hat{x}(T) = \int_{t_{0}}^{T} f(\mathbb{E}\left[ \theta \right],x,t) dt

.. ipython::

    In [1]: f, axarr = plt.subplots(1,3)

    In [1]: for i in range(3):
       ...:     axarr[i].plot(t,Ymean[:,i] - solutionReference[:,i])

    @savefig stochastic_param_compare.png
    In [1]: plt.show()

    In [1]: plt.close()

and we repeat the process with the number of simulation increased

.. ipython::

    In [1]: Ymean, Yall = odeS.simulateParam(t[1::], 1000, full_output=True)

    In [1]: f, axarr = plt.subplots(1, 3)

    In [1]: for i in range(3):
       ...:     axarr[i].plot(t,Ymean[:,i] - solutionReference[:,i])

    @savefig stochastic_param_compare_large_n.png
    In [1]: plt.show()

    In [1]: plt.close()

Obviously, there may be scenarios where only some of the parameters are stochastic.  Let's say that the :math:`\gamma` parameter is fixed at :math:`1/3`, then simply replace the distribution information with a scalar.  A quick look at the resulting plot reveals that it has less variation when compared to the case where both parameters are stochastic.

.. ipython::

    In [1]: d['gamma'] = 1.0/3.0

    In [1]: odeS.setParameters(d)

    In [1]: YmeanSingle, YallSingle = odeS.simulateParam(t[1::], 10, full_output=True)

    In [1]: f, axarr = plt.subplots(1,3)

    In [1]: for solution in YallSingle:
       ...:     axarr[0].plot(t,solution[:,0])
       ...:     axarr[1].plot(t,solution[:,1])
       ...:     axarr[2].plot(t,solution[:,2])

    @savefig stochastic_param_single.png
    In [1]: plt.show()

    In [1]: plt.close()

Continuous Markov Representation
================================

Another common method of introducing stochasticity into a set of ode is by assuming each movement in the system is a result of a jump process.  More concretely, the probabilty of a move for transition :math:`j` is governed by an exponential distribution such that

.. math::

    \Pr(\textnormal{process $j$ jump within time } \tau) = \lambda_{j} e^{-\lambda_{j} \tau},

where :math:`\lambda_{j}` is the rate of transition for process :math:`j` and :math:`\tau` the time elapsed after current time :math:`t`.

A couple of the commmon implementation for the jump process have been implemented where two of them are used during a normal simulation; the first reaction method [1] and the :math:`\tau`-Leap method [2].  The two changes interactively depending on the size of the states.

.. ipython::

    In [1]: x0 = [2362206.0, 3.0, 0.0]

    In [1]: stateList = ['S', 'I', 'R']

    In [1]: paramList = ['beta', 'gamma', 'N']

    In [1]: transitionList = [
       ...:                   Transition(origState='S', destState='I', equation='beta*S*I/N', transitionType=TransitionType.T),
       ...:                   Transition(origState='I', destState='R', equation='gamma*I', transitionType=TransitionType.T)
       ...:                   ]

    In [1]: odeS = SimulateOdeModel(stateList, paramList, transitionList=transitionList)

    In [1]: odeS.setParameters([0.5, 1.0/3.0, x0[0]]).setInitialValue(x0, t[0])

    In [1]: solutionReference = odeS.integrate(t[1::])

    In [1]: simX,simT = odeS.simulateJump(t[1:10], 10, full_output=True)

    In [1]: f, axarr = plt.subplots(1, 3)

    In [1]: for i in range(len(simX)):
       ...:     solution = simX[i]
       ...:     axarr[0].plot(t[:9], solution[:,0])
       ...:     axarr[1].plot(t[:9], solution[:,1])
       ...:     axarr[2].plot(t[:9], solution[:,2])

    @savefig stochastic_process.png
    In [1]: plt.show()

    In [1]: plt.close()

Above, we see ten different simulation, again using the SIR model but the initial conditions are not standardized.  The time frame is shortened to 10 so that the individual changes can be seen more clearly.

.. ipython::

    In [1]: simX, simT = odeS.simulateJump(t, 10, full_output=True)

    In [1]: simMean = numpy.mean(simX, axis=0)

    In [1]: f,axarr = plt.subplots(1, 3)

    In [1]: for i in range(3):
       ...:     axarr[i].plot(t,simMean[:,i] - solutionReference[:,i])

    @savefig stochastic_process_compare.png
    In [1]: plt.show()

    In [1]: plt.close()

same as above, we increase the number of simulation and plot the difference

.. ipython::

    In [1]: simX,simT = odeS.simulateJump(t, 5, full_output=True)

    In [1]: simMean = numpy.mean(simX, axis=0)

    In [1]: f, axarr = plt.subplots(1,3)

    In [1]: for i in range(3):
       ...:     axarr[i].plot(t,simMean[:,i] - solutionReference[:,i])

    @savefig stochastic_process_compare_large_n.png
    In [1]: plt.show()

    In [1]: plt.close()

The difference is significantly greater than the previous assumption, where we assume that the parameters follow some distribution.  The reason behind this is that the jump process above was able to fully remove all the initial infected individuals before any new ones.  Plotting all the epidemiology curve above makes this obvious, note the horizontal lines at the top of the leftmost figure

.. ipython::

    In [1]: f,axarr = plt.subplots(1,3)

    In [1]: for i in range(len(simX)):
       ...:     solution = simX[i]
       ...:     axarr[0].plot(t, solution[:,0])
       ...:     axarr[1].plot(t, solution[:,1])
       ...:     axarr[2].plot(t, solution[:,2])

    @savefig stochastic_process_compare_large_n_curves.png
    In [1]: plt.show()

    In [1]: plt.close()

**Reference**

[1] Exact stochastic simulation of coupled chemical reactions, Gillespie, Danial T., The Journal of Physical Chemistry, Volume 81, Issue 25, pg. 2340-2361, 1977

[2] Efficient step size selection for the tau-leaping simulation method, Cao et el.,  The Journal of Chemical Physics, Volume 124, Issue 4, pg. 044109, 2006
