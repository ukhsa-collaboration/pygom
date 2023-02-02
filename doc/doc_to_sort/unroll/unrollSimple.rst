.. _unrollSimple:

Simple Problem
==============

For a simple problem, we consider the SIR model defined by

.. math::

  \frac{dS}{dt} &= -\beta SI \\
  \frac{dI}{dt} &= \beta SI - \gamma I \\
  \frac{dR}{dt} &= \gamma I.

which consists of two transitions

.. graphviz::

	digraph SIR_Model {
		rankdir=LR;
		size="8"
		node [shape = circle];
		S -> I [ label = "&beta;SI" ];
		I -> R [ label = "&gamma;I" ];
	}

Let's define this using the code block below

.. ipython::

    In [1]: from pygom import SimulateOde, Transition, TransitionType

    In [2]: ode1 = Transition(origin='S', equation='-beta*S*I', transition_type=TransitionType.ODE)

    In [3]: ode2 = Transition(origin='I', equation='beta*S*I - gamma*I', transition_type=TransitionType.ODE)

    In [4]: ode3 = Transition(origin='R', equation='gamma*I', transition_type=TransitionType.ODE)

    In [6]: stateList = ['S', 'I', 'R']

    In [7]: paramList = ['beta', 'gamma']

    In [8]: ode = SimulateOde(stateList,
       ...:                   paramList,
       ...:                   ode=[ode1, ode2, ode3])

    In [9]: ode.get_transition_matrix()

and the last line shows that the transition matrix is empty.  This is the expected result because :class:`SimulateOdeModel` was not initialized using transitions.  We populate the transition matrix below and demonstrate the difference. 

.. ipython::

    In [1]: ode = ode.get_unrolled_obj()

    In [2]: ode.get_transition_matrix()
