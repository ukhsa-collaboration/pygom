===============================
pygom - ODE modelling in Python
===============================

Development of a generic framework for ode models, more specifically, 
aiming to solve compartmental type problems.  

This package depends on::

    enum34
    graphviz
    numpy
    matplotlib
    scipy
    sympy  

and they should be installed if not already available.  Standard installation can be
performed via::

$ python setup.py install

and tested via::

$ python setup.py test

If the test fails, then the documentation for the package will not compile.  A more in depth documentation can be found in the folder::

$ doc

Note that building the documentation can be extremely slow depending on the setup of the system.  Further details can be found at it's own read me::

$ doc/README.rst     

Please be aware that there are currently redundant files within 
the package.

Contributors
============
Edwin Tye (Edwin.Tye@phe.gov.uk)

Version
=======
0.1.0 First exposure
