.. _epi:

************************
Simple Epidemic Analysis
************************

A common application of ordinary differential equations is in the field of epidemiology modeling.  More concretely, compartmental models that is used to describe disease progression.  We demonstrate some of the simple algebraic analysis one may wish to take when given a compartment model.  Our use one of the simplest model, an SIR model with birth and death processes, which is an extension of the one in :ref:`sir`.  First, we initialize the model below.

.. ipython::

    In [1]: from pygom import common_models

    In [2]: ode = common_models.SIR_Birth_Death()

    In [3]: print(ode.get_ode_eqn())


Obtaining the R0
================

The reproduction number, also known as the :math:`R_{0}`, is the single most powerful piece and reduced piece of information available from a compartmental model.  In a nutshell, it provides a single number - if the parameters are known - which can the intuitive interpretation where :math:`R_{0} = 1` defines the tipping point of an outbreak.  A :math:`R_{0}` value of more than one signifies an potential outbreak where less than one indicates that the disease will stop spreading naturally.  

To obtain the :math:`R_{0}`, we simply have to tell the function which states represent the *disease state*, which in this case is the state **I**.

.. ipython::

    In [1]: from pygom.model.epi_analysis import *
    
    In [2]: print(R0(ode, 'I'))

Algebraic R0
============

We may also wish to get the :math:`R_{0}` in pure algebraic term.  This can be achieved by the following few lines.  Note that the result below is slightly different from the one above.  The difference is due to the internal working of the functions, where :func:`getR0` computes the disease-free equilibrium value for the states and substitute them back into the equation.  

.. ipython::

    In [1]: F, V = disease_progression_matrices(ode, 'I')

    In [2]: e = R0_from_matrix(F, V)

    In [3]: print(e)


To replicate the output before, we have to find the values where the disease-free equilibrium will be achieved.  Substitution can then be performed to retrieve :math:`R_{0}` in pure parameters.

.. ipython::

    In [1]: dfe = DFE(ode, ['I'])

    In [2]: print(dfe)

    In [3]: print(e[0].subs(dfe))

