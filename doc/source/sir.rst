.. _sir:

*****************************
Motivating Example: SIR Model
*****************************

Defining the model
==================

First, we are going to go through an SIR model to show the functionality of the package.  The SIR model is defined by the following equations

.. math::

  \frac{dS}{dt} &= -\beta SI \\
  \frac{dI}{dt} &= \beta SI- \gamma I \\
  \frac{dR}{dt} &= \gamma I.

We can set this up as follows

.. ipython:: 

    In [32]: # first we import the classes require to define the transitions

    In [33]: from pygom import Transition, TransitionType

    In [34]: # define our state

    In [35]: stateList = ['S', 'I', 'R']

    In [36]: # and the set of parameters, which only has two

    In [37]: paramList = ['beta', 'gamma']

    In [38]: # then the set of ode

    In [38]: odeList = [
       ....:     Transition(origin='S', equation='-beta*S*I', transition_type=TransitionType.ODE),
       ....:     Transition(origin='I', equation='beta*S*I - gamma*I', transition_type=TransitionType.ODE),
       ....:     Transition(origin='R', equation='gamma*I', transition_type=TransitionType.ODE)
       ....: ]

Here, we have invoke a class from :mod:`Transition` to define the transition object.  We proceed here and ignore the details for now.  The details of defining a transition object will be covered later in :ref:`transition`.  Both the set of states and parameters should be defined when constructing the object, even though not explicitly enforced, to help clarify what we are trying to model.  Similarly, this holds for the rest, such as the derived parameters and transitions, where we force the end user to input the different type of transition/process via the corret argument.  See :ref:`defining-eqn` for an example when the input is wrong.

.. ipython:: 

    In [39]: # now we import the ode module

    In [40]: from pygom import DeterministicOde

    In [41]: # initialize the model

    In [42]: model = DeterministicOde(stateList,
       ....:                          paramList,
       ....:                          ode=odeList)

That is all the information required to define a simple SIR model.  We can verify the equations by

.. ipython::

    In [40]: model.get_ode_eqn()

where we can see the equations corresponding to their respective :math:`S,I` and :math:`R` state. The set of ode is in the standard :math:`S,I,R` sequence because of how the states are defined initially.  We can change them around

.. ipython::

    In [59]: # now we are going to define in a different order.  note that the output ode changed with the input state

    In [60]: stateList = ['R', 'S', 'I']

    In [61]: model = DeterministicOde(stateList,
       ....:                          paramList,
       ....:                          ode=odeList)

    In [62]: model.get_ode_eqn()

and find that the set of ode's still comes out in the correct order with respect to how the states are ordered.  In addition to showing the ode in English, we can also display it as either symbols or latex code which save some extra typing when porting the equations to a proper document.

.. ipython::

    In [1]: model.print_ode()

    In [2]: model.print_ode(True)

The SIR model above was defined as a set of explicit ODEs.  An alternative way is to define the model using a series of transitions between the states.  We have provided the capability to obtain a *best guess* transition matrix when only the ODEs are available.  See the section :ref:`unrollOde` for more information, and in particular :ref:`unrollSimple` for the continuing demonstration of the SIR model.


Model information
=================

The most obvious thing information we wish to know about an ode is whether it is linear

.. ipython:: 

    In [65]: model.linear_ode()

which we know is not for an SIR.  So we may want to have a look at the Jacobian say, it is as simple as 

.. ipython::

    In [64]: model.get_jacobian_eqn()

or maybe we want to know the gradient (of the ode)

.. ipython::

    In [65]: model.get_grad_eqn()

Invoking the functions that computes :math:`f(x)` (or the derivatives) like below will output an error (not run)

.. ipython::

    In [66]: # model.ode()
    
    In [67]: # model.jacobian()

This is because the some of the functions are used to solve the ode numerically and expect input values of both state and time.  But just invoking the two methods above without defining the parameter value, such as the second line below, will also throws an error.

.. ipython::

    In [77]: initialState = [0, 1, 1.27e-6]
    
    In [78]: # model.ode(state=initialState, t=1)

It is important to note at this point that the numeric values of the states need to be set in the correct order against the list of states, which can be found by

.. ipython::
    
    In [79]: model.state_list

There is currently no mechanism to set the numeric values of the states along with the state.  This is because of implementation issue with external package, such as solving an initial value problem.   

Initial value problem
=====================

Setting the parameters will allow us to evaluate

.. ipython::

    In [80]: # define the parameters

    In [81]: paramEval = [
       ....:     ('beta',0.5), 
       ....:     ('gamma',1.0/3.0)
       ....:     ]

    In [82]: model.parameters = paramEval

    In [83]: model.ode(initialState, 1)

Now we are well equipped with solving an initial value problem, using standard numerical integrator such as :func:`odeint <scipy.integrate.odeint>` from :mod:`scipy.integrate`.  We also used :mod:`matplotlib.pyplot` for plotting and :func:`linspace <numpy.linspace>` to create the time vector.

.. ipython::

    In [96]: import scipy.integrate

    In [97]: import numpy

    In [98]: t = numpy.linspace(0, 150, 100)

    In [99]: solution = scipy.integrate.odeint(model.ode, initialState, t)

    In [100]: import matplotlib.pyplot as plt

    In [101]: plt.figure();

    In [102]: plt.plot(t, solution[:,0], label='R');

    In [103]: plt.plot(t, solution[:,1], label='S');

    In [104]: plt.plot(t, solution[:,2], label='I');

    In [105]: plt.xlabel('Time');

    In [106]: plt.ylabel('Population proportion');

    In [107]: plt.title('Standard SIR model');
    
    In [108]: plt.legend(loc=0);
    
    @savefig sir_plot.png
    In [109]: plt.show();

    In [110]: plt.close()

Where a nice standard SIR progression can be observed in the figure above.  Alternatively, we can also integrate and plot via the **ode** object which we have initialized.  

.. ipython::

    In [1]: model.initial_values = (initialState, t[0])

    In [2]: model.parameters = paramEval

    In [3]: solution = model.integrate(t[1::])

    In [4]: model.plot()

The plot is not shown as it is identical to the one above without the axis labels.  Obviously, we can solve the ode above using the Jacobian as well.  Unfortunately, it does not help because the number of times the Jacobian was evaluated was zero, as expected given that our set of equations are not stiff.

.. ipython::

    In [583]: %timeit solution1, output1 = scipy.integrate.odeint(model.ode, initialState, t, full_output=True)

    In [584]: %timeit solution2, output2 = scipy.integrate.odeint(model.ode, initialState, t, Dfun=model.jacobian, mu=None, ml=None, full_output=True)

    In [584]: %timeit solution3, output3 = model.integrate(t, full_output=True)

It is important to note that we return our Jacobian as a dense square matrix.  Hence, the two argument (mu,ml) for the ode solver was set to ``None`` to let it know the output explicitly.

Solving the forward sensitivity equation
========================================

Likewise, the sensitivity equations are also solved as an initial value problem.  Let us redefine the model in the standard SIR order and we solve it with the sensitivity all set at zero, i.e. we do not wish to infer the initial value of the states

.. ipython::

    In [452]: stateList = ['S', 'I', 'R']

    In [453]: model = DeterministicOde(stateList,
       .....:                          paramList,
       .....:                          ode=odeList)

    In [454]: initialState = [1, 1.27e-6, 0]

    In [455]: paramEval = [
       .....:              ('beta', 0.5), 
       .....:              ('gamma', 1.0/3.0)
       .....:             ]

    In [456]: model.parameters = paramEval

    In [457]: solution = scipy.integrate.odeint(model.ode_and_sensitivity, numpy.append(initialState, numpy.zeros(6)), t)

    In [458]: f,axarr = plt.subplots(3,3);

    In [459]: # f.text(0.5,0.975,'SIR with forward sensitivity solved via ode',fontsize=16,horizontalalignment='center',verticalalignment='top');

    In [460]: axarr[0,0].plot(t, solution[:,0]);

    In [461]: axarr[0,0].set_title('S');

    In [462]: axarr[0,1].plot(t, solution[:,1]);

    In [463]: axarr[0,1].set_title('I');

    In [464]: axarr[0,2].plot(t, solution[:,2]);

    In [465]: axarr[0,2].set_title('R');

    In [466]: axarr[1,0].plot(t, solution[:,3]);

    In [467]: axarr[1,0].set_title(r'state S parameter $\beta$');

    In [468]: axarr[2,0].plot(t, solution[:,4]);

    In [469]: axarr[2,0].set_title(r'state S parameter $\gamma$');

    In [470]: axarr[1,1].plot(t, solution[:,5]);

    In [471]: axarr[1,1].set_title(r'state I parameter $\beta$');

    In [472]: axarr[2,1].plot(t, solution[:,6]);

    In [473]: axarr[2,1].set_title(r'state I parameter $\gamma$');

    In [474]: axarr[1,2].plot(t, solution[:,7]);

    In [475]: axarr[1,2].set_title(r'state R parameter $\beta$');

    In [476]: axarr[2,2].plot(t, solution[:,8]);

    In [477]: axarr[2,2].set_title(r'state R parameter $\gamma$');

    In [478]: plt.tight_layout();
    
    @savefig sir_sensitivity_plot.png
    In [480]: plt.show();

    In [481]: plt.close()

This concludes the introductory example and we will be moving on to look at parameter estimation next in :ref:`estimate1` and the most important part in terms of setting up the ode object; defining the equations in various different ways in :ref:`transition`.

