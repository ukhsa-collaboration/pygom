
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
