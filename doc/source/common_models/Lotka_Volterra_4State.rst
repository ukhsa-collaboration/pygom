:func:`.Lotka_Volterra_4State`
==============================

The Lotka-Volterra model with four states and three parameters, explained by the following three transitions

.. math::

    \frac{da}{dt} &= k_{0} a x \\
    \frac{dx}{dt} &= k_{0} a x - k_{1} x y \\
    \frac{dy}{dt} &= k_{1} x y - k_{2} y \\
    \frac{db}{dt} &= k_{2} y.

First, we show the deterministic approach.  Then we also show the different process path using the parameters from [2].  Note that although the model is defined in :mod:`common_models`, it is based on outputting an :class:`OperateOdeModel` rather than :class:`SimulateOdeModel`.

.. ipython::

    In [1]: import matplotlib.pyplot as plt
    
    In [1]: from pygom import Transition, TransitionType, ode_utils, SimulateOdeModel

    In [1]: import numpy

    In [1]: stateList = ['a','x','y','b']
    
    In [1]: paramList = ['k0','k1','k2']

    In [1]: transitionList = [
       ...:                   Transition(origState='a',destState='x',equation='k0 * a * x',transitionType=TransitionType.T),
       ...:                   Transition(origState='x',destState='y',equation='k1 * x * y',transitionType=TransitionType.T),
       ...:                   Transition(origState='y',destState='b',equation='k2 * y',transitionType=TransitionType.T)
       ...:                   ]

    In [1]: ode = SimulateOdeModel(stateList, paramList,transitionList=transitionList)

    In [1]: x0 = [150.0, 10.0, 10.0, 0.0]
    
    In [1]: t = numpy.linspace(0,15,100)

    In [1]: solution = ode.setInitialValue(x0,t[0]).setParameters([0.01,0.1,1.0]).integrate(t[1::])

    @savefig common_models_Lotka_Volterra_4State.png
    In [1]: ode.plot()

    In [1]: simX,simT = ode.simulateJump(t[1::],5,full_output=True)

    In [1]: f,axarr = plt.subplots(1,3)
    
    In [1]: for i in range(0,len(simX)):
       ...:     solution = simX[i]
       ...:     axarr[0].plot(simT,solution[:,0])
       ...:     axarr[1].plot(simT,solution[:,1])
       ...:     axarr[2].plot(simT,solution[:,2])

    @savefig common_models_Lotka_Volterra_Sim.png    
    In [1]: plt.show()

**References**

[1] Analytical Note on Certain Rhythmic Relations in Organic Systems, Lotka Alfred J., Proceedings of the National Academy of Sciences of the United States of America, Volume 7, Issue 7, pg. 410-415, 1920.

[2] Numerical Recepies 3rd Edition: The Art of Scientific Computing, Press et al., Cambridge University Press, 2007.
