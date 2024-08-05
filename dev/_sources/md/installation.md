# Installation

Installation instructions may be found on the [GitHub project README](https://github.com/ukhsa-collaboration/pygom/), but we include them here also.

## From source

Source code for PyGOM can be downloaded from the GitHub repository: https://github.com/ukhsa-collaboration/pygom

```bash
git clone https://github.com/ukhsa-collaboration/pygom.git
```

Please be aware that there may be redundant files within the package as it is under active development.

```{note}
The latest fully reviewed version of PyGOM will be on the master branch and we recommend that users install the version from there.
```

Activate the relevant branch for installation via Git Bash:

```bash
git activate relevant-branch-name
```

Package dependencies can be found in the file `requirements.txt`.
An easy way to install these is to create a new [conda](https://conda.io/docs) environment via:

```bash
conda env create -f conda-env.yml
```

which you should ensure is active for the installation process using:

```bash
conda activate pygom
```

Alternatively, you may add dependencies to your own environment.

```bash
pip install -r requirements.txt
```

The final prerequisite, if you are working on a Windows machine, is that you will also need to install:
- [Graphviz](https://graphviz.org/)
- Microsoft Visual C++ 14.0 or greater, which you can get with [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

You you should be able to install the PyGOM package via command line:

```bash
python setup.py install
```

If you anticipate making your own frequent changes to the PyGOM source files, it might be more convenient to install in develop mode instead:

```bash
python setup.py develop
```

## From PyPI

Alternatively, the latest release can be installed from [PyPI](https://pypi.org/project/pygom/):

```bash
pip install pygom
```

# Testing the package

Test files should then be run from the command line to check that installation has completed successfully

```bash
python setup.py test
```

This can take some minutes to complete.
