# Pygom workflow

The PyGom workflow is as follows

1. Encapsulate the ODE system using a series of {class}`Transition` objects.
2. Feed this into a class {class}`DeterministicOde` or {class}`SimulateOde` depending on how solutions are required.
3. Use functionality to verify or update model specification. Find solutions etc.

