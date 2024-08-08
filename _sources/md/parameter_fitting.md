# Parameter fitting

As well as producing forecasts, another key activity in infectious disease modelling is inference of epidemic parameters from case data.
In this chapter we outline how PyGOM may be used to assist these endeavours.
In {doc}`the first section <../notebooks/paramfit/params_via_abc>`, we present a more up to date method which uses Approximate Bayesian Computation (ABC) and {doc}`then <../notebooks/paramfit/params_via_optimization>` a more classical approach via Maximum Likelihood Estimation (MLE).
We also demonstrate PyGOM's ability to solve the less epidemiologically related task of {doc}`boundary value problems <../notebooks/paramfit/bvpSimple>`