# Set up

<!---
## What does this package do?

The purpose of this package is to allow the end user to easily define a
set of ordinary differential equations (ODEs) and obtain information
about the ODEs by invoking the the appropriate methods. Here, we define
the set of ODEs as

$$\frac{\mathrm{d} \mathbf{y}}{\mathrm{d} t} = f(\mathbf{y},\boldsymbol{\theta})$$

where $\mathbf{y} = \left(y_{1},y_{2},\ldots,y_{n}\right)$ is the state vector and $\boldsymbol{\theta} = \left(\theta_{1},\theta_{2},\ldots, \theta_{p}\right)$ the set of parameters. Currently, this package allows the user to find the algebraic
expression of the ODE, Jacobian, gradient and forward sensitivity of the
ODE. A numerical output is given when all the state and parameter values
are provided. Note that the only important class is
{class}`.DeterministicOde` where all the
functionality described previously are exposed.

Plans for further development can be found, and proposed, on the repository's [issue board](https://github.com/ukhsa-collaboration/pygom/issues).
-->

## Installation

### From source

Source code for PyGOM can be downloaded from the GitHub repository: https://github.com/ukhsa-collaboration/pygom

```bash
git clone https://github.com/ukhsa-collaboration/pygom.git
```

```{note}
The latest fully reviewed version of PyGOM will be on the master branch and we recommend that users install the version from there.
```

Dependencies may be added by creating an environment using conda

```bash
conda env create -f conda-env.yml
```

This environment should be active during the installation process

```bash
conda activate pygom
```

Alternatively, you may add dependencies to your own environment.

```bash
pip install -r requirements.txt
```

If you are working on a Windows machine you will also need to install:
- [Graphviz](https://graphviz.org/)
- [Visual C++](https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0)
- [Visual C++ Build Tools](https://go.microsoft.com/fwlink/?LinkId=691126)

You you should be able to install the PyGOM package via command line:

```bash
python setup.py install
```

If you anticipate making your own frequent changes to the PyGOM source files, it might be more convenient to install in develop mode instead:

```bash
python setup.py develop
```

### From PyPI

Alternatively, the latest release can be installed from [PyPI](https://pypi.org/project/pygom/):

```bash
pip install pygom
```

```{note}
Please note that there are some redundant files that are being kept for development purposes.
```

## Testing the package

Test files can be run from the command line to check that installation has completed successfully

```bash
python setup.py test
```

## Building the documentation locally

The documentation which you are currently reading may be built locally.
First, install additional packages:

```bash
pip install -r docs/requirements.txt
```

Then build the documentation from command line

```bash
jupyter-book build docs
```

The html files will be saved in the local copy of your repository under:

    pygom/docs/_build/html