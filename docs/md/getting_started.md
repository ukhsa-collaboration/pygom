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
{class}`.DeterministicOde` where all the
functionality described previously are exposed.

Plans for further development can be found, and proposed, on the repository's [issue board](https://github.com/ukhsa-collaboration/pygom/issues).

## Installing the package 

PyGOM can be downloaded from the GitHub repository.

https://github.com/PublicHealthEngland/pygom.git

You will need to create an environment, for example using conda. 

    conda env create -f conda-env.yml

Alternatively, add dependencies to your own environment.

    pip install -r requirements.txt

If you are working on a Windows machine you will also need to install:
- [Graphviz](https://graphviz.org/)
- [Visual C++](https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0)
- [Visual C++ Build Tools](https://go.microsoft.com/fwlink/?LinkId=691126)

You can install the package via command line:

    python setup.py install

or locally on a user level:

    python setup.py install --user

```{note}
The latest fully reviewed version of PyGOM will be on master branch. We recommend that users install the version from this branch.
```

Alternatively the latest release can be installed from [PyPI](https://pypi.org/project/pygom/):

    pip install pygom

Please note that there are some redundant files that are being kept for
development purposes.

## Testing the package

Test files can be run from the command line prior to or after installation.

    python setup.py test

## Building the documentation locally

Install additional packages:

    pip install -r docs/requirements.txt

Build the documentation:

    jupyter-book build docs/

The html files will be saved in the local copy of your repository under:

    pygom/docs/_build/html


## Using this documentation
This documentation is built using [JupyterBook](https://jupyterbook.org/en/stable/intro.html). To use the contents of a notebook as a starting point for trialing or developing your own models and analyses, you can download any of the examples within this documentation by using the download icon on the desired page (located at the top right).

![download file](../images/download.png)

## Contributing to PyGOM

Please see the [contribution guidance](../../CONTRIBUTING.md) which outlines:
- required information for raising issues;
- the process by which code contributions should be incorporated;
- what is required by pull requests to PyGOM, including how to add to the documentation;
- how we will acknowledge your contributions.