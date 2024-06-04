# PyGOM - Python Generic ODE Model

[![pypi version](https://img.shields.io/pypi/v/pygom.svg)](https://pypi.python.org/pypi/pygom)
[![Documentation Status](https://readthedocs.org/projects/pygom/badge/?version=master)](https://pygom.readthedocs.io/en/master/?badge=master)
[![licence](https://img.shields.io/pypi/l/pygom?color=green)](https://raw.githubusercontent.com/PublicHealthEngland/pygom/master/LICENSE.txt)
[![Github actions](https://github.com/PublicHealthEngland/pygom/workflows/pygom/badge.svg)](https://github.com/PublicHealthEngland/pygom/actions/)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://hwilliams-phe.github.io/pygom/intro.html)

A generic framework for Ordinary Differential Equation (ODE) models, especially compartmental type systems.
This package provides a simple interface for users to construct ODE models backed by a comprehensive and easy to use tool–box implementing functions to easily perform common operations such as parameter estimation and solving for deterministic or stochastic time evolution.
With both the algebraic and numeric calculations performed automatically (but still accessible),
the end user is free to focus on model development.

## Installation

You can download a local copy of the PyGOM source files from this GitHub repository:

    git clone https://github.com/ukhsa-collaboration/pygom.git

Please be aware that there may be redundant files within the package as it is under active development.

> [!NOTE]
> The latest fully reviewed version of PyGOM will be on the `master` branch and we generally recommend that users install this version. However, the current version being prepared for release (featuring up-to-date documentation) is hosted on the `feature/prep-doc` branch.

Activate the relevant branch for installation via Git Bash, if you have been recruited to test code for the
new release then this is the `feature/prep-doc` branch:

    git activate feature/prep-doc

Package dependencies can be found in the file, `requirements.txt`.
An easy way to install these to create a new [conda](https://conda.io/docs) environment via:

    conda env create -f conda-env.yml

which you should ensure is active for the installation process using

    conda activate pygom

Alternatively, you may add dependencies to your own environment:

    pip install -r requirements.txt

The final prerequisite, if you are working on a Windows machine, is that you will also need to install:
- [Graphviz](https://graphviz.org/)
- [Visual C++](https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0)
- [Visual C++ Build Tools](https://go.microsoft.com/fwlink/?LinkId=691126)

You should now be able to install the PyGOM package via command line:

    python setup.py install

and test that installation has completed successfully

    python setup.py test

This will run a few test cases and can take some minutes to complete.

## Documentation

Documentation must be built locally and all necessary files can be found in the `docs` folder.
Documentation is built from the command line:

    jupyter-book build docs

The html files will be saved in the local copy of your repository under:

    docs/_build/html

> [!NOTE]
> Building the documentation involves running many examples in python which can take up to tens of minutes. Subsequent builds with these examples unchanged are much quicker due to caching of the code outputs.

Please be aware that if the module tests fails, then the documentation for the package will not compile.

## Contributors

Thomas Finnie (Thomas.Finnie@phe.gov.uk)

Edwin Tye

Hannah Williams

Jonty Carruthers

Martin Grunnill

Joseph Gibson

## Version

0.1.7 Add Approximate Bayesian Computation (ABC) as a method of fitting to data 

0.1.6 Bugfix scipy API, pickling, print to logging and simulation

0.1.5 Remove auto-simplification for much faster startup

0.1.4 Much faster Tau leap for stochastic simulations

0.1.3 Defaults to python built-in unittest and more in sync with conda
