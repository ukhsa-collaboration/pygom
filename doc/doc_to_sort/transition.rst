.. _transition:

*****************
Transition Object
*****************

The most important part of setting up the model is to correctly define the set odes, which is based solely on the classes defined in :mod:`transition`.  All transitions that gets fed into the ode system needs to be defined as a transition object, :class:`Transition`.  It takes a total of four input arguments

#. The origin state
#. Equation that describe the process
#. The type of transition
#. The destination state

where the first three are mandatory.  To demonstrate, we go back to the SIR model defined previously in the section :ref:`sir`.  Recall that the set of odes are

.. math::

  \frac{\partial S}{\partial t} &= -\beta SI \\
  \frac{\partial I}{\partial t} &= \beta SI - \gamma I \\
  \frac{\partial R}{\partial t} &= \gamma I.

We can simply define the set of ode, as seen previously, via

.. ipython::

    In [1]: from pygom import Transition, TransitionType, common_models

    In [2]: ode1 = Transition(origin='S', equation='-beta*S*I', transition_type=TransitionType.ODE)

    In [3]: ode2 = Transition(origin='I', equation='beta*S*I - gamma*I', transition_type=TransitionType.ODE)

    In [4]: ode3 = Transition(origin='R', equation='gamma*I', transition_type=TransitionType.ODE)

Note that we need to state explicitly the type of equation we are inputting, which is simply of type **ODE** in this case.  We can confirm this has been entered correctly by putting it into :class:`DeterministicOde`

.. ipython::

    In [1]: from pygom import DeterministicOde

    In [2]: stateList = ['S', 'I', 'R']

    In [3]: paramList = ['beta', 'gamma']

    In [4]: model = DeterministicOde(stateList,
       ...:                          paramList,
       ...:                          ode=[ode1, ode2, ode3])

and check it 

.. ipython::

    In [1]: model.get_ode_eqn()

An alternative print function :func:`print_ode` is also available which may be more suitable in other situation.  The default prints the formula in a rendered format and another which prints out the latex format which can be used directly in a latex document.  The latter is useful as it saves typing out the formula twice, once in the code and another in documents.

.. ipython::

    In [1]: model.print_ode(False)

    In [2]: model.print_ode(True)

Now we are going to show the different ways of defining the same set of odes.

.. _defining-eqn:

Defining the equations
======================

Recognizing that the set of odes defining the SIR model is the result of two transitions,

.. math::

    S \rightarrow I &= \beta SI \\
    I \rightarrow R &= \gamma I

where :math:`S \rightarrow I` denotes a transition from state :math:`S` to state :math:`I`.  Therefore, we can simply define our model by these two transition, but now these two transition needs to be inputted via the ``transition`` argument instead of the ``ode`` argument.  Note that we are initializing the model using a different class, because the stochastic implementation has more operation on transitions.

.. ipython::

    In [600]: from pygom import SimulateOde
    
    In [601]: t1 = Transition(origin='S', destination='I', equation='beta*S*I', transition_type=TransitionType.T)

    In [602]: t2 = Transition(origin='I', destination='R', equation='gamma*I', transition_type=TransitionType.T)

    In [603]: modelTrans = SimulateOde(stateList,
       .....:                          paramList,
       .....:                          transition=[t1, t2])

    In [604]: modelTrans.get_ode_eqn()

We can see that the resulting ode is exactly the same, as expected.  The transition matrix that defines this process can easily be visualized using graphviz.  Because only certain renderer permit the use of sub and superscript, operators such as :math:`**` are left as they are in the equation.  

.. ipython::

    In [1]: import matplotlib.pyplot as plt

    In [2]: f = plt.figure()

    In [3]: modelTrans.get_transition_matrix()
    
    @savefig sir_transition_graph.png
    In [4]: dot = modelTrans.get_transition_graph()

If we put in via the wrong argument like below (not run), then an error will appear.

.. ipython::

    In [1]: # modelTrans = DeterministicOde(stateList, paramList, ode=[t1, t2])

because :class:`TranstionType` was defined explicitly as a transition instead of an ode.  The same can be observed when the wrong :class:`TransitionType` is used for any of the input argument.

This though, only encourages us to define the transitions carefully.  We can also pretend that the set of odes are in fact just a set of birth process

.. ipython::

    In [619]: birth1 = Transition(origin='S', equation='-beta*S*I', transition_type=TransitionType.B)

    In [620]: birth2 = Transition(origin='I', equation='beta*S*I - gamma*I', transition_type=TransitionType.B)

    In [621]: birth3 = Transition(origin='R', equation='gamma*I', transition_type=TransitionType.B)

    In [622]: modelBirth = DeterministicOde(stateList,
       .....:                               paramList,
       .....:                               birth_death=[birth1, birth2, birth3])

    In [623]: modelBirth.get_ode_eqn()

which will yield the same set result.  Alternatively, we can use the negative of the equation but set it to be a death process.  For example, we multiply the equations for state :math:`S` and :math:`R` with a negative sign and set the transition type to be a death process instead.

.. ipython::

    In [624]: death1 = Transition(origin='S', equation='beta*S*I', transition_type=TransitionType.D)

    In [625]: birth2 = Transition(origin='I', equation='beta*S*I - gamma*I', transition_type=TransitionType.B)

    In [626]: death3 = Transition(origin='R', equation='-gamma*I', transition_type=TransitionType.D)

    In [627]: modelBD = DeterministicOde(stateList,
       .....:                            paramList,
       .....:                            birth_death=[death1, birth2, death3])

    In [628]: modelBD.get_ode_eqn()


We can see that all the above ways yield the same set of ode at the end.

Model Addition
==============

Because we allow the separation of transitions between states and birth/death processes, the birth/death processes can be added later on.  

.. ipython::

    In [1]: modelBD2 = modelTrans
    
    In [1]: modelBD2.param_list = paramList + ['mu', 'B']
    
    In [1]: birthDeathList = [Transition(origin='S', equation='B', transition_type=TransitionType.B),
       ...:                   Transition(origin='S', equation='mu*S', transition_type=TransitionType.D),
       ...:                   Transition(origin='I', equation='mu*I', transition_type=TransitionType.D)]
    
    In [1]: modelBD2.birth_death_list = birthDeathList

    In [1]: modelBD2.get_ode_eqn()
    
So modeling can be done in stages.  Start with a standard closed system and extend it with additional flows that interact with the environment.

.. _transition-type:

Transition type
===============

There are currently four different type of transitions allowed, which is defined in an enum class also located in :mod:`transition`.  The four types are B, D, ODE and T, where they represent different type of process with explanation in their corresponding value.

.. ipython::

    In [1]: from pygom import transition

    In [2]: for i in transition.TransitionType:
       ...:     print(str(i) + " = " + i.value)
	   
Each birth process are added to the origin state while each death process are deducted from the state, i.e. added to the state after multiplying with a negative sign.  An ode type is also added to the state and we forbid the number of input ode to be greater than the number of state inputted.
