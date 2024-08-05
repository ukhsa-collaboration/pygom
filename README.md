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

## Installation
The easiest way to install a copy of PyGOM is via PyPI and pip
    
    pip install pygom

Alternatively, you can download a local copy of the PyGOM source files from this GitHub repository:

    git clone https://github.com/ukhsa-collaboration/pygom.git

Please be aware that there may be redundant files within the package as it is under active development.

> [!NOTE]
> The latest fully reviewed version of PyGOM will be on the `master` branch and we generally recommend 
> that users install this version. However, the latest version being prepared for release is hosted on 
> the `dev` branch.

When running the following command line commands, ensure that your current working directory is the one 
where the PyGOM source files were downloaded to. This should be found from your home directory:

    cd pygom

Activate the relevant branch for installation via Git Bash. for example if you want
new release then this is the `dev` branch:

    git checkout dev

Package dependencies can be found in the file, `requirements.txt`.
An easy way to install these to create a new [conda](https://conda.io/docs) environment in Anaconda Prompt via:

    conda env create -f conda-env.yml

which you should ensure is active for the installation process using

    conda activate pygom

Alternatively, you may add dependencies to your own environment through conda:

    conda install --file requirements.txt

**or** via pip:

    pip install -r requirements.txt

The final prerequisites, if you are working on a Windows machine, is that you will also need to install:
- [Graphviz](https://graphviz.org/)
- Microsoft Visual C++ 14.0 or greater, which you can get with [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

You should now be able to install the PyGOM package via command line:

    pip install .

and test that installation has completed successfully

    python -m unittest discover --verbose --start-directory tests

This will run a few test cases and can take some minutes to complete.

## Documentation

Documentation must be built locally and all necessary files can be found in the `docs` folder.
Documentation is built from the command line by first installing the additional documentation requirements:

    pip install -r docs/requirements.txt

and then building the documentation:

    jupyter-book build docs

The html files will be saved in the local copy of your repository under:

    docs/_build/html

You can view the documentation by opening the index file in your browser of choice:

    docs/_build/html/index.html

> [!NOTE]
> Building the documentation involves running many examples in python which can take up to 30 minutes. Subsequent builds with these examples unchanged are much quicker due to caching of the code outputs.

Please be aware that if the module tests fails, then the documentation for the package will not compile.

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
