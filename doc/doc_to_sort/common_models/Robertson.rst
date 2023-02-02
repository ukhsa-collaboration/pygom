:func:`.Robertson`
==================

The Robertson problem [Robertson1966]_

.. math::
    
    \frac{dy_{1}}{dt} &= -0.04 y_{1} + 1 \cdot 10^{4} y_{2} y_{3} \\
    \frac{dy_{2}}{dt} &= 0.04 y_{1} - 1 \cdot 10^{4} y_{2} y_{3} + 3 \cdot 10^{7} y_{2}^{2} \\
    \frac{dy_{3}}{dt} &= 3 \cdot 10^{7} y_{2}^{2}.
    
This is a problem that describes an autocatalytic reaction.  One of those commonly used to test stiff ode solvers.  As the parameters in the literature is fixed, we show here how to define the states in a slightly more compact format

.. ipython::

    In [1]: from pygom import DeterministicOde, Transition, TransitionType
    
    In [1]: import numpy
    
    In [1]: import matplotlib.pyplot as plt
    
    In [1]: t = numpy.append(0, 4*numpy.logspace(-6, 6, 1000))

    In [1]: # note how we define the states    

    In [1]: stateList = ['y1:4']

    In [1]: paramList = []
    
    In [1]: transitionList = [
       ...:                  Transition(origin='y1', destination='y2', equation='0.04*y1', transition_type=TransitionType.T),
       ...:                  Transition(origin='y2', destination='y1', equation='1e4*y2*y3', transition_type=TransitionType.T),
       ...:                  Transition(origin='y2', destination='y3', equation='3e7*y2*y2', transition_type=TransitionType.T)
       ...:                  ]

    In [1]: ode = DeterministicOde(stateList, paramList, transition=transitionList)

    In [1]: ode.initial_values = ([1.0, 0.0, 0.0], t[0])

    In [1]: solution, output = ode.integrate(t[1::], full_output=True)

    In [1]: f, axarr = plt.subplots(1, 3)
    
    In [1]: for i in range(3):
       ...:     axarr[i].plot(t, solution[:,i])
       ...:     axarr[i].set_xscale('log')

    In [1]: f.tight_layout();

    @savefig common_models_Robertson_1.png
    In [1]: plt.show()

    In [1]: plt.close()

To simplify even further, we can use `y` with the corresponding subscript directly instead of `y1,y2,y3`.  Again, we do not have any parameters as they are hard coded into our models.

.. ipython::

    In [1]: stateList = ['y1:4']

    In [1]: transitionList = [
       ...:                  Transition(origin='y[0]', destination='y[1]', equation='0.04*y[0]', transition_type=TransitionType.T),
       ...:                  Transition(origin='y[1]', destination='y[0]', equation='1e4*y[1]*y[2]', transition_type=TransitionType.T),
       ...:                  Transition(origin='y[1]', destination='y[2]', equation='3e7*y[1]*y[1]', transition_type=TransitionType.T)
       ...:                  ]

    In [1]: ode = DeterministicOde(stateList, paramList, transition=transitionList)

    In [1]: ode.initial_values =([1.0, 0.0, 0.0], t[0])

    In [1]: solution2 = ode.integrate(t[1::])

    In [1]: numpy.max(solution - solution2)

and we have the identical solution as shown in the last line above.
