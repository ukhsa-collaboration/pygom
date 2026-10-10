# PyGOM - Python Generic ODE Model

[![pypi version](https://img.shields.io/pypi/v/pygom.svg)](https://pypi.python.org/pypi/pygom)
[![licence](https://img.shields.io/pypi/l/pygom?color=green)](https://raw.githubusercontent.com/ukhsa-collaboration/pygom/master/LICENSE.txt)
[![Github actions](https://github.com/ukhsa-collaboration/pygom/workflows/pygom/badge.svg)](https://github.com/ukhsa-collaboration/pygom/actions/)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](http://ukhsa-collaboration.github.io/pygom/md/intro.html)

A generic framework for Ordinary Differential Equation (ODE) models, especially compartmental type systems.
This package provides a simple interface for users to construct ODE models backed by a comprehensive and easy to use tool–box implementing functions to easily perform common operations such as parameter estimation and solving for deterministic or stochastic time evolution.
With both the algebraic and numeric calculations performed automatically (but still accessible),
the end user is free to focus on model development.
Full documentation for this package is avalible on the [documentation](http://ukhsa-collaboration.github.io/pygom/md/intro.html) page.

# Installation

## Install from Source

The PyGOM source code is available from GitHub:

<https://github.com/ukhsa-collaboration/pygom>

### Clone the repository

```bash
git clone https://github.com/ukhsa-collaboration/pygom.git
```

### Create and activate an environment

If you require support for compartmental model diagrams, create (and activate) a Conda environment with Graphviz installed:

```bash
conda create -n pygom-env python=3.11 graphviz
conda activate pygom-env
```

If visualisation is not required, you may use a standard Conda environment or a Python virtual environment (venv).

### Install PyGOM

Navigate to the project root

```bash
cd pygom
```

If you wish to install from a specific branch, switch to it prior to installation:

```bash
git checkout <branch-name>
```

Install PyGOM

```bash
python -m pip install .
```

For development work, an editable install can instead be used

```bash
python -m pip install -e .
```

To build the documentation locally, install the additional documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

```{note}
Please be aware that there may be redundant files within the package as it is under active development.
The latest fully reviewed version of PyGOM will be on the master branch and we recommend that users install the version from there.
```

## Install from PyPI

Alternatively, install the latest released version directly from PyPI:

<https://pypi.org/project/pygom/>

```bash
pip install pygom
```

## Verifying the Installation

Run the test suite to confirm that the installation completed successfully:

```bash
python -m unittest discover --verbose --start-directory tests
```

This may take some minutes to complete.


## Contributors

Thomas Finnie (Thomas.Finnie@ukhsa.gov.uk)

Edwin Tye

Hannah Williams

Jonty Carruthers

Martin Grunnill

Joseph Gibson

## Version
0.1.8 Updated and much better documentation.

0.1.7 Add Approximate Bayesian Computation (ABC) as a method of fitting to data 

0.1.6 Bugfix scipy API, pickling, print to logging and simulation

0.1.5 Remove auto-simplification for much faster startup

0.1.4 Much faster Tau leap for stochastic simulations

0.1.3 Defaults to python built-in unittest and more in sync with conda
