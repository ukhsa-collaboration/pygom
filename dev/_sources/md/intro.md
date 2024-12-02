# Welcome to the documentation for PyGOM

## What does this package do?

PyGOM (Python Generic ODE Model) is a Python package which provides a simple interface for users to construct Ordinary Differential Equation (ODE) models, with a focus on compartmental models and epidemiology.
This is backed by a comprehensive and easy to use tool–box implementing functions to easily perform common operations such as parameter estimation and solving for deterministic or stochastic time evolution.
The package source is freely available (hosted on [GitHub](https://github.com/ukhsa-collaboration/pygom)) and organized in a way that permits easy extension. With both the algebraic and numeric calculations performed automatically (but still accessible), the end user is freed to focus on model development.

## Release notes:

### [0.1.9] - 2024-11-30 (Latest release)

- Added
    - Method {func}`solve_stochast` of {class}`SimulateOde` has 1 additional output (for a total of 3): the number of times each event occurs in between each time step. This is useful if you are interested in infection incidence rather than prevalence, for example.
    - The adaptive tau leap can be bypassed and, instead, a constant step size for the tau leap algorithm can be specified with `SimulateOde._pre_tau`.
    - When an output of a stochastic simulation needs to be mapped to the user defined time-points, this is performed via linear interpolation.
- Changed
    - The {class}`Event` class has replaced the {class}`Transition` class as the fundamental building block. This allows for more flexibility when dealing with correlated transitions, for example.
    - Internal methods {func}`add_func` and {func}`add_compiled_sympy_object` make the compilation of sympy objects more modular.
- Deprecated
    - Models can still be defined via {class}`Transition` objects as before, but users are advised to switch to the {class}`Event` based approach. 
    - The birth rate state can still be defined as an `origin`, but can now also (and more accurately) be described as a `destination`. In the next version, this may remove the need to specify the transition type since between state transitions will uniquely have an origin and destination, births will uniquely have just a destination and deaths just an origin.
- Removed
- Fixed
    - Minor bug fixes

### [0.1.8] - 2024-08-06

- Added
    - Comprehensive documentation of how to use PyGOM
- Changed
    - Running simulations with random parameters does not require a special simulation function. Instead, PyGOM now recognises the parameter types handed to it (fixed or random) and acts accordingly. This means that stochastic simulations can now be performed with random parameters.
- Deprecated
    - {class}`DeterministicOde` is deprecated with {class}`SimulateOde` now performing both deterministic and stochastic simulations.
- Removed
- Fixed
    - Minor bug fixes

## Using this documentation
This documentation is built using [JupyterBook](https://jupyterbook.org/en/stable/intro.html).
Instructions on how to build the documentation locally and where to find it can be found {doc}`here <building_doc>`.
To use the contents of a notebook as a starting point for trialling or developing your own models and analyses, you can download any of the examples within this documentation by using the download icon on the desired page (located at the top right).

![download file](../images/download.png)

## Contributing to PyGOM

Please see the [contribution guidance](https://github.com/ukhsa-collaboration/pygom/blob/master/CONTRIBUTING.md) which outlines:
- Required information for raising issues
- The process by which code contributions should be incorporated
- What is required by pull requests to PyGOM, including how to add to the documentation
- How we will acknowledge your contributions
