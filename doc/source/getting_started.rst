.. _getting_started:

***************
Getting started
***************

.. _package-purpose:

What this package does
======================

The purpose of this package is to allow the end user to easily define a set of
ordinary differential equations (ode) and obtain information about the ode by
simply invoking the the appropriate methods.  Here, we define the set of ode's
as

.. math::
    \frac{\partial \mathbf{x}}{\partial t} = f(\mathbf{x},\boldsymbol{\theta})

where :math:`\mathbf{x} = \left(x_{1},x_{2},\ldots,x_{n}\right)` is the state
vector with :math:`d` state and :math:`\boldsymbol{\theta}` the parameters of
:math:`p` dimension.  Currently, this package allows the end user to find the
algebraic expression of the ode, Jacobian, gradient and forward sensitivity of
the ode.  A numerical output is given when all the state and parameter values
are provided.   Note that the only important class is :file:`DeterministicOde`
all the functionality described previously are exposed.

The current plan is to extend the functionality to include

* Solving the ode analytically when it is linear

* Analysis of the system via eigenvalues during the integration

* Detection of DAE


.. _installing-docdir:

Obtaining the package
=====================

The location of the package is current on GitHub and can be pulled via https
from::

    https://github.com/PublicHealthEngland/pygom.git

The package is currently as follows::

  pygom/
      bin/
      doc/
      pygom/
          loss/
              tests/
          model/
              tests/
          sbml_translate/
          utilR/
      LICENSE.txt
      README.rst
      requirements.txt
      setup.py

with files in each of the three main folder not shown.  You can install the
package via command line::

    python setup.py install

or locally on a user level::

    python setup.py install --user

Please note that there are current redundant file are kept for development
purposes for the time being.

.. _testing-the-package:

Testing the package
===================

Testing can be performed prior or after the installation.  Some standard test
files can be found in their respective folder and they can be run in the command
line::

    python setup.py test

which can be performed prior to installing the package if desired.
