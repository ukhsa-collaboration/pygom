.. _faq:

************************
Frequent asked questions
************************

Code runs slowly
================

This is because the package is not optimized for speed.  Although the some of the main functions are lambdified using :mod:`sympy` or compiled against :mod:`cython` when available, there are many more optimization that can be done.  One example is the lines:

.. python:

    J = self.Jacobian(state,t)
    G = self.Grad(state,t)
    A = numpy.dot(J,S) + G
    
in :func:`.DeterministicOde.evalSensitivity`.  The first two operations can be inlined into the third and the third line itself can be rewritten as:

.. python:

    G += numpy.dot(J,S)
	
and save the explicit copy operation by :mod:`numpy` when making A. If desired, we could have also made used of the :mod:`numexpr` package that provides further speed up on elementwise operations in place of numpy.

Why not compile the numeric computation form sympy against Theano
=================================================================

Setup of the package has been simplified as much as possible.  If you look closely enough, you will realize that the current code generation only uses :mod:`cython` and not :mod:`f2py`.  This is because we are not prepared to do all the system checks, i.e. does a fortran compiler exist, is gcc installed, was python built as a shared library etc.  We are very much aware of the benefit, especially considering the possibility of GPU computation in :mod:`theano`.

Why not use mpmath library throughout?
======================================

This is because we have a fair number of operations that depends on :mod:`scipy`.  Obviously, we can solve ode using :mod:`mpmath` and do standard linear algebra.  Unfortunately, optimization and statistics packages and routine are mostly based on :mod:`numpy`.

Computing the gradient using :class:`.SquareLoss` is slow
=========================================================

It will always be slow on the first operation.  This is due to the design where the initialization of the class is fast and only find derivative information/compile function during runtime.  After the first calculation, things should be significantly faster.

**Why some of my code is not a fortran object?**

When we detec either a :math:`\exp` or a :math:`\log` in the equations, we automatically force the compile to use mpmath to ensure that we obtain the highest precision.  To turn this on/off will be considered as a feature in the future.

Can you not convert a non-autonumous system to an autonomous system for me automatically
========================================================================================

Although we can do that, it is not, and will not be implemented.  This is to ensure that the end user such as yourself are fully aware of the equations being defined.

Getting the sensitivities from :class:`.SquareLoss` did not get a speed up when I used a restricted set of parameters
=====================================================================================================================

This is because we currently evaluate the full set of sensitivities before extracting them out.  Speeding this up for a restrictive set is being considered.  A main reason that stopped us from implementing is that we find the symbolic gradient of the ode before compiling it.  Which means that one function call to the compiled file will return the full set of sensitivities and we would only be extracting the appropriate elements from the matrix.  This only amounts to a small speed up.  The best method would be to compile only the necessary elements of the gradient matrix, but this would require much more work both within the code, and later on when variables are being added/deleted as all these compilation are perfromed in runtime.

Why do not have the option to obtain gradient via complex differencing
======================================================================

It is currently not implemented.  Feature under consideration.


