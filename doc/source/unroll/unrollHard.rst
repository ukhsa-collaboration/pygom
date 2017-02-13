.. _unrollHard:

Hard Problem
============

Now we turn to a harder problem that does not have a one to one mapping between all the transitions and the terms in the ODEs.  We use the model in :func:`Influenza_SLIARN`, defined by 

.. math::
    \frac{dS}{dt} &= -S \beta (I + \delta A) \\    
    \frac{dL}{dt} &= S \beta (I + \delta A) - \kappa L \\  
    \frac{dI}{dt} &= p \kappa L - \alpha I \\
    \frac{dA}{dt} &= (1-p) \kappa L - \eta A \\
    \frac{dR}{dt} &= f \alpha I + \eta A \\
    \frac{dN}{dt} &= -(1-f) \alpha I. 

The outflow of state **L**, :math:`\kappa L`, is composed of two transitions, one to **I** and the other to **A** but the ode of **L** only reflects the total flow going out of the state.  Same can be said for state **I**, where the flow :math:`\alpha I` goes to both **R** and **N**.

We slightly change the model by introducing a new state **D** to convert it into a closed system.  The combination of state **D** and **N** is a constant, the total population.  So we can remove **N** and this new system consist of six transitions.  We define them explicitly as ODEs and unroll them into transitions.

.. ipython::

    In [1]: from pygom import SimulateOdeModel, Transition, TransitionType

    In [1]: stateList = ['S', 'L', 'I', 'A', 'R', 'D']

    In [2]: paramList = ['beta','p','kappa','alpha','f','delta','epsilon', 'N']

    In [3]: odeList = [
       ...:            Transition(origState='S', equation='- beta * S/N * ( I + delta * A)', transitionType=TransitionType.ODE),
       ...:            Transition(origState='L', equation='beta * S/N * (I + delta * A) - kappa * L', transitionType=TransitionType.ODE),
       ...:            Transition(origState='I', equation='p * kappa * L - alpha * I', transitionType=TransitionType.ODE),
       ...:            Transition(origState='A', equation='(1-p) * kappa * L - epsilon * A', transitionType=TransitionType.ODE),
       ...:            Transition(origState='R', equation='f * alpha * I + epsilon * A', transitionType=TransitionType.ODE),
       ...:            Transition(origState='D', equation='(1-f) * alpha * I', transitionType=TransitionType.ODE) ]

    In [4]: ode = SimulateOdeModel(stateList, paramList, odeList=odeList)

    In [5]: ode.getTransitionMatrix()

    In [6]: ode2 = ode.returnObjWithTransitionsAndBD()

    In [7]: ode2.getTransitionMatrix()
    
    In [8]: ode2.getOde()

After unrolling the odes, we have the following transition graph

.. graphviz::

	digraph SLIARD_Unroll {
	    labelloc = "t";
	    label = "Unrolled transitions";
		rankdir=LR;
		size="8"
		node [shape = circle];
		S -> L [ label = "-S&beta;(I + &delta;A)/N" ];
		L -> A [ label = "&kappa;L" ];
		A -> I [ label = "&kappa;Lp" ];
		I -> D [ label = "&alpha;I" ];
		A -> R [ label = "&eta;A" ];
		D -> R [ label = "&alpha;If" ];
	}

which obviously combines to the same set of ODE, but the underlying mechanism differs significantly when compared to the original transition graph.

.. graphviz::

	digraph SLIARD_Model {
		labelloc = "t";
	    label = "Original transitions";
		rankdir=LR;
		size="8"
		node [shape = circle];
		S -> L [ label = "-S&beta;(I + &delta;A)/N" ];
		L -> I [ label = "&kappa;Lp" ];
		L -> A [ label = "&kappa;L(1-p)" ];
		I -> R [ label = "&alpha;If" ];
		I -> D [ label = "&alpha;I(1-f)" ];
		A -> R [ label = "&eta;A" ];
	}


The most notable difference is that this new representation only has a single absorbed state in **R** rather than two, **R** and **D**.  It also means that **D** can decrease which implies that people can rise from the dead, a mechanism not yet possible with current technology.  

As mentioned previously, the set of ODEs is the same, which means that they have the same deterministic solution.  However, the simulated result under a jump process differs.  

.. ipython::

    In [1]: import numpy

    In [1]: import matplotlib.pyplot as plt

    In [1]: t = numpy.linspace(0, 50, 200)
    
    In [2]: x0 = [2000.0, 0.0, 12.0, 0.0, 0.0, 0.0]

    In [3]: params = {'kappa':0.526,
       ...:           'alpha':0.244,
       ...:           'epsilon':0.255,
       ...:           'p':2.0/3.0,
       ...:           'delta':0.5,
       ...:           'f':0.98,
       ...:           'beta':3.0,
       ...:           'N': 2012.0}

    In [4]: solution2 = ode2.setParameters(params).setInitialValue(x0,t[0]).integrate2(t[1::])

    In [5]: simX2, simT2 = ode2.simulateJump(t, 5, full_output=True)

    In [6]: f,axarr = plt.subplots(2,3)
    
    In [7]: for solution in simX2:
       ...:     for i in range(3):
       ...:         axarr[0,i].plot(t,solution[:,i])
       ...:         axarr[0,i].set_title(str(ode.getStateList()[i]))
       ...:         axarr[1,i].plot(t,solution[:,i+3])
       ...:         axarr[1,i].set_title(str(ode.getStateList()[i+3]))

    @savefig sliarn_ctmc_unroll.png
    In [8]: plt.show()

    In [9]: plt.close()

compare to the solution path using the correct model transitions

.. ipython::

    In [1]: transitionList2 = [Transition(origState='S', destState='L', equation='beta * S/N * ( I + delta * A)', transitionType=TransitionType.T),
       ...:                    Transition(origState='L', destState='A', equation='L * kappa * (1-p)', transitionType=TransitionType.T),
       ...:                    Transition(origState='L', destState='I', equation='L * kappa * p', transitionType=TransitionType.T),
       ...:                    Transition(origState='A', destState='R', equation='A * epsilon', transitionType=TransitionType.T),
       ...:                    Transition(origState='I', destState='D', equation='alpha * I * (1-f)', transitionType=TransitionType.T),
       ...:                    Transition(origState='I', destState='R', equation='alpha * I * f', transitionType=TransitionType.T)]

    In [3]: ode = SimulateOdeModel(stateList, paramList, transitionList=transitionList2)

    In [4]: solution = ode.setParameters(params).setInitialValue(x0,t[0]).integrate(t[1::])

    In [5]: simX, simT = ode.simulateJump(t, 5, full_output=True)

    In [6]: f,axarr = plt.subplots(2,3)

    In [7]: for solution in simX:
       ...:     for i in range(3):
       ...:         axarr[0,i].plot(t,solution[:,i])
       ...:         axarr[0,i].set_title(str(ode.getStateList()[i]))
       ...:         axarr[1,i].plot(t,solution[:,i+3])
       ...:         axarr[1,i].set_title(str(ode.getStateList()[i+3]))
       
    @savefig sliarn_ctmc_orig.png
    In [8]: plt.show()

    In [9]: plt.close()

The difference can be seen in the plots.  Most notably the path taken by the **D** state, with the unrolled transition being able to go up and down.  Unless we have discovered a way to revive the dead, it is fair to say that the model using the unrolled transitions is not (remotely close to) an accurate representation.
