:func:`.SEIR_Multiple`
=========================

Multiple SEIR coupled together, without any birth death process.

.. math::

    \frac{dS_{i}}{dt} &= dN_{i} - dS_{i} - \lambda_{i}S_{i} \\
    \frac{dE_{i}}{dt} &= \lambda_{i}S_{i} - (d+\epsilon)E_{i} \\
    \frac{dI_{i}}{dt} &= \epsilon E_{i} - (d+\gamma) I_{i} \\
    \frac{dR_{i}}{dt} &= \gamma I_{i} - dR_{i}

where

.. math::
    \lambda_{i} = \sum_{j=1}^{n} \beta_{i,j} I_{j} (1\{i\neq j\} p)
    
with :math:`n` being the number of patch and :math:`p` the coupled factor.

.. ipython::
     
    In [0]: from pygom import common_models
    
    In [1]: paramEval = {'beta_00':0.0010107,'beta_01':0.0010107,'beta_10':0.0010107,
       ...:              'beta_11':0.0010107,'d':0.02,'epsilon':45.6,'gamma':73.0,
       ...:              'N_0':10**6,'N_1':10**6,'p':0.01}
    
    In [2]: x0 = [36139.3224081278, 422.560577637822, 263.883351688369, 963174.233662546]
    
    In [3]: ode = common_models.SEIR_Multiple()
    
    In [4]: t = numpy.linspace(0,40,100)
    
    In [5]: x01 = []
    
    In [6]: for s in x0:
       ...:     x01 += [s]
       ...:     x01 += [s]
    
    In [7]: ode.setParameters(paramEval).setInitialValue(numpy.array(x01,float),t[0])
    
    In [8]: solution,output = ode.integrate(t[1::],full_output=True)

    @savefig common_models_seir_multiple.png
    In [9]: ode.plot()

The initial conditions are those derived by using the stability condition in [1] while the notations is taken from [2].

**References**

[1] Lloyd A.L. and May R.M., Spatial Heterogeneity in Epidemic Models, Journal of Theoretical Biology, Vol 179, no. 1, pg 1-11, 1996

[2] Mathematical Epidemiology, Lecture Notes in Mathematics, Brauer Fred, Springer 2008
