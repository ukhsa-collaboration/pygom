===============================
pygom - ODE modelling in Python
===============================

|pypi version|  |Build status|

.. |pypi version| image:: https://img.shields.io/pypi/v/pygom.svg
   :target: https://pypi.python.org/pypi/pygom
.. |Build status| image:: https://travis-ci.org/PublicHealthEngland/pygom.svg?branch=master
   :target: https://travis-ci.org/PublicHealthEngland/pygom
.. |Docs| image:: https://readthedocs.org/projects/pygom/badge/?version=master
   :target: https://pygom.readthedocs.io/en/master/?badge=master
   :alt: Documentation Status

Development of a generic framework for ode models, more specifically,
aiming to solve compartmental type problems.

This package depends on::

    dask
    matplotlib
    enum34
    pandas
    python-dateutil
    python-libsbml
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

If the test fails, then the documentation for the package will not compile.  A more in depth documentation can be found in the folder::

$ doc

Note that building the documentation can be extremely slow depending on the setup of the system.  Further details can be found at it's own read me::

$ doc/README.rst

Please be aware that there may be redundant files within the package as it is under active development.

Contributors
============
Edwin Tye (Edwin.Tye@phe.gov.uk)

Version
=======
0.1.3 Defaults to python built-in unittest and more in sync with conda
