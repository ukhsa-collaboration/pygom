=======================
Documentation for pygom
=======================

This documentation is written in `sphinx` style.  The starting point (about the
package) can be found at::

    $ doc/_build/html/index.html

which is the main page in html format. 

If you wish to build the documentation.   Go to the document directory and type 
the following line in the terminal::

    doc$ make html

It is assumed here that the pyGenericOdeModel package has already 
been installed, else, the code demonstration will throw out errors
as it cannot find the required modules. Most likely, building the
documentation will require the following packages if not already installed::

    numpydoc
    sphinx
    ipython


There may be cases that `ipython` require extra packages, install the full 
version using `$ pip install ipython[all]` 

============================
Installation of requirements
============================

If you are using `conda` the requirements to build the docs can be installed 
by `$ conda install --file requirements.txt` or similarly for those using `pip` 
you may `$pip install -r requirements.txt` from the root of the docs directory.