===============================
pygom - ODE modelling in Python
===============================

|Github actions|  |Documentation Status|  |pypi version|  |licence|

.. |pypi version| image:: https://img.shields.io/pypi/v/pygom.svg
   :target: https://pypi.python.org/pypi/pygom
.. |Documentation Status| image:: https://readthedocs.org/projects/pygom/badge/?version=master
   :target: https://pygom.readthedocs.io/en/master/?badge=master
.. |licence| image:: https://img.shields.io/pypi/l/pygom?color=green   :alt: PyPI - License
   :target: https://raw.githubusercontent.com/PublicHealthEngland/pygom/master/LICENSE.txt
.. |Github actions| image:: https://github.com/PublicHealthEngland/pygom/workflows/pygom/badge.svg
   :target: https://github.com/PublicHealthEngland/pygom/actions/

A generic framework for ode models, specifically compartmental type problems.

This package depends on::

    dask
    matplotlib
    enum34
    pandas
    python-dateutil
    numpy
    scipy
    sympy

and they should be installed if not already available.  Alternatively, the easier way
to use a minimal (and isolated) setup is to use `conda <https://conda.io/docs/>`_ and
create a new environment via::

  conda env create -f conda-env.yml

Installation of this package can be performed via::

$ python setup.py install

and tested via::

$ python setup.py test

A reduced form of the documentation may be found on ReadTheDocs_.

.. _ReadTheDocs: https://pygom.readthedocs.io/en/master/

You may get the full documentation, including the lengthy examples by locally
building the documentation found in the folder::

$ doc

Note that building the documentation can be extremely slow depending on the
setup of the system.  Further details can be found at it's own read me::

$ doc/README.rst

Please be aware that if the module tests fails, then the documentation for the
package will not compile.

Please be aware that there may be redundant files within the package as it is
under active development.

Contributors
============
Thomas Finnie (Thomas.Finnie@phe.gov.uk)

Edwin Tye

Hannah Williams

Jonty Carruthers

Martin Grunnill

Version
=======
0.1.7 Add Approximate Bayesian Computation (ABC) as a method of fitting to data 

0.1.6 Bugfix scipy API, pickling, print to logging and simulation

0.1.5 Remove auto-simplification for much faster startup

0.1.4 Much faster Tau leap for stochastic simulations

0.1.3 Defaults to python built-in unittest and more in sync with conda
