# Producing forecasts

Once the system of ODE's has been specified, PyGOM allows the user to numerically solve for both the deterministic and stochastic time evolution.
Furthermore, users may specify model parameters to take either fixed values or to be drawn randomly from a probability distribution.

In this chapter, we will use an SIR model as our example system to introduce

- How to prescribe parameters in {doc}`Parameterisation <../notebooks/model_params>`

- How to obtain solutions and process the model output in {doc}`Finding ODE solutions <../notebooks/model_solver>`