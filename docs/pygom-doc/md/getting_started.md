# Getting started

## What does this package do?

The purpose of this package is to allow the end user to easily define a
set of ordinary differential equations (ODEs) and obtain information
about the ODEs by invoking the the appropriate methods. Here, we define
the set of ODEs as

$$\frac{d \mathbf{x}}{d t} = f(\mathbf{x},\boldsymbol{\theta})$$

where $\mathbf{x} = \left(x_{1},x_{2},\ldots,x_{n}\right)$ is the state
vector with $d$ state and $\boldsymbol{\theta}$ the parameters of $p$
dimension. Currently, this package allows the user to find the algebraic
expression of the ODE, Jacobian, gradient and forward sensitivity of the
ODE. A numerical output is given when all the state and parameter values
are provided. Note that the only important class is
`DeterministicOde`{.interpreted-text role="file"} where all the
functionality described previously are exposed.

#TODO do we want this updating or referencing issue board?
The current plan is to extend the functionality to include

-   Solving the ode analytically when it is linear
-   Analysis of the system via eigenvalues during the integration
-   Detection of DAE

## Obtaining the package 

The location of the package is current on GitHub and can be pulled via
https from:

    https://github.com/PublicHealthEngland/pygom.git

#TODO required?
The package is currently as follows:

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

with files in each of the three main folders not shown. You can install
the package via command line:

    python setup.py install

or locally on a user level:

    python setup.py install --user

Please note that there are current redundant file are kept for
development purposes for the time being.

## Testing the package

Testing can be performed prior or after the installation. Some standard
test files can be found in their respective folder and they can be run
in the command line:

    python setup.py test

which can be performed prior to installing the package if desired.

## Using this documentation
To use a notebook as a starting point, you can download any of the source code within this jupyterbook by using the download icon that is located at the top of each page of the documentation.
![download file](../images/download.png)

## Contributing to PyGOM

Please see the (contribution guidance)[../../CONTRIBUTING.md] which outlines:
- required information for raising issues;
- the process by which code contributions should be incorporated;
- what is required by additions to PyGOM, including how to add to the jupyterbook;
- expectations for acknowledgements from contributions.
