# Welcome to the documentation for PyGOM

## What does this package do?

PyGOM (Python Generic ODE Model) is a Python package which provides a simple interface to easily construct systems
governed by Ordinary Differential Equations (ODEs), with a focus on compartmental models and epidemiology.
This is backed by a comprehensive and easy to use tool–box which implements functions to easily perform common operations for ODE
systems such as solving, parameter estimation, and stochastic simulation.
The package source is freely available (hosted on [Github](https://github.com/PublicHealthEngland/pygom)) and organized in a way that permits easy extension. With both the algebraic and numeric calculations performed automatically (but still accessible), the end user is freed to focus on model development.

## Using this documentation
This documentation is built using [JupyterBook](https://jupyterbook.org/en/stable/intro.html). To use the contents of a notebook as a starting point for trialing or developing your own models and analyses, you can download any of the examples within this documentation by using the download icon on the desired page (located at the top right).

![download file](../images/download.png)

## Contributing to PyGOM

Please see the [contribution guidance](https://github.com/ukhsa-collaboration/pygom/blob/master/CONTRIBUTING.md) which outlines:
- Required information for raising issues
- The process by which code contributions should be incorporated
- What is required by pull requests to PyGOM, including how to add to the documentation
- How we will acknowledge your contributions

<!---
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
-->

```{tableofcontents}
```
