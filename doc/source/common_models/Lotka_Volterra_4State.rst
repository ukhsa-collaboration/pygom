:func:`.Lotka_Volterra_4State`
==============================

The Lotka-Volterra model with four states and three parameters [Lotka1920]_, explained by the following three transitions

.. math::

    \frac{da}{dt} &= k_{0} a x \\
    \frac{dx}{dt} &= k_{0} a x - k_{1} x y \\
    \frac{dy}{dt} &= k_{1} x y - k_{2} y \\
    \frac{db}{dt} &= k_{2} y.

First, we show the deterministic approach.  Then we also show the different process path using the parameters from [Press2007]_.  Note that although the model is defined in :mod:`common_models`, it is based on outputting an :class:`OperateOdeModel` rather than :class:`SimulateOdeModel`.

.. ipython::

    In [1]: import matplotlib.pyplot as plt

    In [1]: from pygom import Transition, TransitionType, ode_utils, SimulateOde

    In [1]: import numpy

    In [1]: stateList = ['a', 'x', 'y', 'b']

    In [1]: paramList = ['k0', 'k1', 'k2']

    In [1]: transitionList = [
       ...:                   Transition(origin='a', destination='x', equation='k0*a*x', transition_type=TransitionType.T),
       ...:                   Transition(origin='x', destination='y', equation='k1*x*y', transition_type=TransitionType.T),
       ...:                   Transition(origin='y', destination='b', equation='k2*y', transition_type=TransitionType.T)
       ...:                   ]

    In [1]: ode = SimulateOde(stateList, paramList, transition=transitionList)

    In [1]: x0 = [150.0, 10.0, 10.0, 0.0]

    In [1]: t = numpy.linspace(0, 15, 100)

    In [1]: ode.initial_values = (x0, t[0])

    In [1]: ode.parameters = [0.01, 0.1, 1.0]

    In [1]: solution = ode.integrate(t[1::])

    @savefig common_models_Lotka_Volterra_4State.png
    In [1]: ode.plot()

    In [1]: simX, simT = ode.simulate_jump(t[1::], 5, full_output=True)

    @savefig common_models_Lotka_Volterra_Sim.png
    In [1]: ode.plot(simX, simT)

