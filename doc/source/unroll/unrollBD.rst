.. _unrollBD:

ODE With Birth and Death Process
================================

We follow on from the SIR model of :ref:`unrollSimple` but with additional birth and death processes.

.. math::

  \frac{dS}{dt} &= -\beta SI + B - \mu S\\
  \frac{dI}{dt} &= \beta SI- \gamma I - \mu I\\
  \frac{dR}{dt} &= \gamma I.

which consists of two transitions and three birth and death process

.. graphviz::

	digraph SIR_Model {
		rankdir=LR;
		size="8"
		node [shape = circle];
		S -> I [ label = "&beta;SI" ];
		I -> R [ label = "&gamma;I" ];
        B [height=0 margin=0 shape=plaintext width=0];
        B -> S;
        "S**2*&mu;" [height=0 margin=0 shape=plaintext width=0];
        S -> "S**2*&mu;";
        "I*&mu;" [height=0 margin=0 shape=plaintext width=0];
		I -> "I*&mu;";
	}

Let's define this in terms of ODEs, and unroll it back to the individual processes.

.. ipython::

    In [1]: from pygom import Transition, TransitionType, SimulateOdeModel, common_models
    
    In [1]: stateList = ['S', 'I', 'R']

    In [1]: paramList = ['beta', 'gamma', 'B', 'mu']

    In [1]: odeList = [
       ...:            Transition(origState='S', 
       ...:                       equation='-beta * S * I + B - mu * S',
       ...:                       transitionType=TransitionType.ODE),
       ...:            Transition(origState='I', 
       ...:                       equation='beta * S * I - gamma * I - mu * I',
       ...:                       transitionType=TransitionType.ODE),
       ...:            Transition(origState='R', 
       ...:                       equation='gamma * I',
       ...:                       transitionType=TransitionType.ODE)
       ...:            ]

    In [1]: ode = SimulateOdeModel(stateList, paramList, odeList=odeList)
    
    In [1]: ode2 = ode.returnObjWithTransitionsAndBD()
    
    @savefig sir_unrolled_transition_graph.png
    In [1]: ode2.getTransitionGraph()
    
    In [1]: plt.close()
    