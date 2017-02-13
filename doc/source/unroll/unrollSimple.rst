.. _unrollSimple:

Simple Problem
==============

For a simple problem, we consider the SIR model defined by

.. math::

  \frac{dS}{dt} &= -\beta SI \\
  \frac{dI}{dt} &= \beta SI- \gamma I \\
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

Let's define this using the ode 

.. ipython::

    In [1]: from pygom import SimulateOdeModel, Transition, TransitionType

    In [2]: ode1 = Transition(origState='S',equation='-beta*S*I', transitionType=TransitionType.ODE)

    In [3]: ode2 = Transition(origState='I',equation='beta*S*I - gamma * I', transitionType=TransitionType.ODE)

    In [4]: ode3 = Transition(origState='R',equation='gamma*I', transitionType=TransitionType.ODE)

    In [6]: stateList = ['S', 'I', 'R']

    In [7]: paramList = ['beta', 'gamma']

    In [8]: ode = SimulateOdeModel(stateList,
       ...:                        paramList,
       ...:                        odeList=[ode1,ode2,ode3])

    In [9]: ode.getTransitionMatrix()

and the last line shows that the transition matrix is empty.  This is the expected result because :class:`SimulateOdeModel` was not initialized using transitions.

.. ipython::

    In [1]: ode = ode.returnObjWithTransitionsAndBD()

    In [2]: ode.getTransitionMatrix()
