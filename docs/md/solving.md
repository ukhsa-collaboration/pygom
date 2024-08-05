# Producing forecasts

An exercise central to the study of infectious diseases (and indeed ODE models in general) is performing simulations to understand the likely evolution of the system in time.
PyGOM allows the user to easily obtain numerical solutions for both the deterministic and stochastic time evolution of their model.
Furthermore, users may specify model parameters to take either fixed values or to be drawn randomly from a probability distribution.

In this chapter, we will use an SIR model as our example system to introduce

- How to prescribe parameters in {doc}`Parameterisation <../notebooks/model_params>`
- How to obtain solutions and process the model output in {doc}`Finding ODE solutions <../notebooks/model_solver>`