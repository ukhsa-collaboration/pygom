.. _estimate2:

*******************************
Example: Parameter Estimation 2
*******************************

Continuing from the :ref:`estimate1`, we show why estimating the parameters for ode's are hard.  This is especially true if there is a lack of data or when there are too much flexibility in the model.  Note that for reproducibility purposes, only deterministic models are used here and a fixed seed whenever a stochastic algorithm is needed.

Standard SEIR model
===================

We demonstrate the estimation on the recent Ebola outbreak in West Africa.   We use the number of deaths in Guinea and its corresponding time the data was recorded.  These data are publicly available and they can be obtained easily on the internet such as https://github.com/cmrivers/ebola.  It is stated out here for simplicity.

.. ipython::

    In [34]: # the number of deaths and cases in Guinea 

    In [35]: yDeath = [29.0, 59.0, 60.0, 62.0, 66.0, 70.0, 70.0, 80.0, 83.0, 86.0, 95.0, 101.0, 106.0, 108.0, 
       ....:           122.0, 129.0, 136.0, 141.0, 143.0, 149.0, 155.0, 157.0, 158.0, 157.0, 171.0, 174.0, 
       ....:           186.0, 193.0, 208.0, 215.0, 226.0, 264.0, 267.0, 270.0, 303.0, 305.0, 307.0, 309.0, 
       ....:           304.0, 310.0, 310.0, 314.0, 319.0, 339.0, 346.0, 358.0, 363.0, 367.0, 373.0, 377.0, 
       ....:           380.0, 394.0, 396.0, 406.0, 430.0, 494.0, 517.0, 557.0, 568.0, 595.0, 601.0, 632.0, 
       ....:           635.0, 648.0, 710.0, 739.0, 768.0, 778.0, 843.0, 862.0, 904.0, 926.0, 997.0]

    In [35]: yCase = [49.0, 86.0, 86.0, 86.0, 103.0, 112.0, 112.0,  122.0,  127.0,  143.0,  151.0,  158.0,  
       ....:          159.0,  168.0,  197.0, 203.0,  208.0,  218.0,  224.0,  226.0,  231.0,  235.0,  236.0,  
       ....:          233.0,  248.0,  258.0,  281.0,  291.0,  328.0,  344.0, 351.0,  398.0,  390.0,  390.0,  
       ....:          413.0,  412.0,  408.0,  409.0,  406.0,  411.0,  410.0,  415.0,  427.0,  460.0,  472.0, 
       ....:          485.0, 495.0, 495.0, 506.0, 510.0, 519.0,  543.0,  579.0, 607.0,  648.0,  771.0,  
       ....:          812.0,  861.0,  899.0,  936.0, 942.0, 1008.0, 1022.0, 1074.0, 1157.0, 1199.0, 1298.0, 
       ....:          1350.0, 1472.0, 1519.0, 1540.0, 1553.0, 1906.0]

    In [36]: # the corresponding time

    In [37]: t = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 13.0, 16.0, 18.0, 20.0, 23.0, 25.0, 26.0, 29.0,
       ....:      32.0, 35.0, 40.0, 42.0, 44.0, 46.0, 49.0, 51.0, 62.0, 66.0, 67.0, 71.0, 73.0, 80.0, 86.0, 88.0, 
       ....:      90.0, 100.0, 102.0, 106.0, 108.0, 112.0, 114.0, 117.0, 120.0, 123.0, 126.0, 129.0, 132.0, 135.0, 
       ....:      137.0, 140.0, 142.0, 144.0, 147.0, 149.0, 151.0, 157.0, 162.0, 167.0, 169.0, 172.0, 175.0, 176.0, 
       ....:      181.0, 183.0, 185.0, 190.0, 193.0, 197.0, 199.0, 204.0, 206.0, 211.0, 213.0, 218.0]

Simple estimation
-----------------

First ,we are going to fit a standard **SEIR** model to the data.  Details of the models can be found in :func:`modelDef.SEIR` Defining the model as usual with some random guess on what the parameters might be, here, we choose the values to be the mid point of our feasible region (defined by our box constraints later)

.. ipython:: 

    In [1]: from pygom import SquareLoss, common_models
    
    In [1]: import numpy, scipy.optimize
    
    In [1]: import matplotlib.pyplot as plt

    In [1]: theta = numpy.array([5.0,5.0,5.0])

    In [2]: ode = common_models.SEIR(theta)

    In [3]: population = 1175e4

    In [4]: y = numpy.reshape(numpy.append(numpy.array(yCase),numpy.array(yDeath)),(len(yCase),2),'F') / population 
    
    In [5]: x0 = [1., 0., 49.0/population, 29.0/population]
    
    In [6]: t0 = t[0]

    In [7]: objLegrand = SquareLoss(theta,ode,x0,t0,t[1::],y[1::,:],['I','R'],[1175e4,1175e4])

Then we optimize, first, assuming that the initial conditions are accurate.  Some relatively large bounds are used for this particular problem.  

.. ipython:: 

    In [8]: boxBounds = [ (0.0,10.0), (0.0,10.0), (0.0,10.0) ]

    In [213]: res = scipy.optimize.minimize(fun=objLegrand.cost,
       .....:                               jac=objLegrand.sensitivity,
       .....:                               x0=theta,
       .....:                               bounds=boxBounds,
       .....:                               method='SLSQP')

    In [9]: print(res)

    @savefig ebola_seir_straight.png
    In [10]: objLegrand.plot()

    In [10]: plt.close()

We can see from our visual confirmation that the estimated parameters are not exactly ideal. This is most likely due to the poor starting point.  An attempt to find a more suitable value can be done by some form of parameter space exploration.  Given that the evaluation of the objective function is not expensive here, we have plenty of options to choose from.  To reduce the number of packages required to build this documentaion, routines from :mod:`scipy.optimize` remains our preferred option.  

Improved initial guess
----------------------

.. ipython::

    In [212]: resDE = scipy.optimize.differential_evolution(objLegrand.cost,bounds=boxBounds,polish=False,seed=20921391)

    In [213]: res = scipy.optimize.minimize(fun=objLegrand.cost,
       .....:                               jac=objLegrand.sensitivity,
       .....:                               x0=resDE['x'],
       .....:                               bounds=boxBounds,
       .....:                               options={'maxiter':1000},
       .....:                               method='SLSQP')

    In [9]: print(res)

    @savefig ebola_seir_de.png
    In [10]: objLegrand.plot()

    In [10]: plt.close()

Looking at the output of the estimates (below this paragraph), we can see our inference on Ebola is completely wrong even though the graphs looks *``reasonable"*.  Namely, :math:`\gamma^{-1}` the third element in the vector below, our time from infectious to death, is much higher than expected while :math:`\alpha^{-1}` (second element), incubation period, is lower than expected.

.. ipython::

    In [1]: 1/res['x']

Multimodal surface
------------------

A reason for this type of behavior is that we simply lack the information/data to make proper inference.  Without data on the state **E**, the parameters :math:`\beta,\alpha` for the two states **I** and **E** are dependent only on observations on **I**.  Hence, some other random combination of :math:`\beta,\alpha` that is capable of generating realization close to observations in **I** is feasible.  In such cases, the only requirement is that there exist some :math:`\gamma` in the feasible region that can compensate for the ill suited :math:`\beta,\alpha`.  For example, we know (obtained elsewhere and not shown here) that there is another set of parameters capable of generating a similar looking curves as before.  Note the reversal of magnitude in :math:`\beta` and :math:`\alpha`.

.. ipython:: 

    In [10]: # we expect the line below to produce something like [3.26106524e+00,   2.24798702e-04,   1.23660721e-02] or [ 0.02701867,  9.00004776,  0.01031861]

    In [10]: print(res['x'])

    In [10]: objLegrand.cost([3.26106524e+00,   2.24798702e-04,   1.23660721e-02]) #ode.setParameters([ 0.02701867,  9.00004776,  0.01031861])

    @savefig ebola_seir_prior.png
    In [10]: objLegrand.plot()

    In [10]: plt.close()

With initial values as parameters
---------------------------------

Obviously, the assumption that the whole population being susceptible is an overestimate.  We now try to estimate the initial conditions of the ode as well.  Given previous estimates of the parameters :math:`\hat{\beta},\hat{\alpha},\hat{\gamma}` it is appropriate to start our initial guess there.  

Furthermore, given that we now estimate the initial values for all the states, we can use the first time point as our observation.  So our time begins at :math:`t = -1` where our observations include the previous initial condition, i.e. 49 and 29 for the number of cases and death at :math:`t=0` respectively.

.. ipython:: 

    In [1]: thetaIV = theta.tolist() + x0

    In [2]: thetaIV[3] -= 1e-8 # to make sure that the initial guess satisfy the constraints

    In [3]: boxBoundsIV = boxBounds + [(0.,1.),(0.,1.),(0.,1.),(0.,1.)]

    In [4]: objLegrand = SquareLoss(theta,ode,x0,-1,t,y,['I','R'],[1175e4,1175e4])

    In [5]: resDEIV = scipy.optimize.differential_evolution(objLegrand.costIV,bounds=boxBoundsIV,polish=False,seed=20921391)

    In [6]: xIV = resDEIV['x']

    In [213]: res = scipy.optimize.minimize(fun=objLegrand.costIV,
       .....:                               jac=objLegrand.sensitivityIV,
       .....:                               x0=resDEIV['x'],
       .....:                               bounds=boxBoundsIV,
       .....:                               options={'maxiter':1000},
       .....:                               method='SLSQP')

    In [18]: # or this line for a refined solution # solution = ode.setInitialValue(res['x'][-4:],-1).setParameters(res['x'][:3]).integrate(t)

    In [19]: print(res)

    @savefig ebola_seir_iv.png
    In [20]: objLegrand.plot()

    In [21]: plt.close()


Legrand Ebola SEIHFR Model
==========================

Next, we demonstrate the estimation on a model that is widely used in the recent Ebola outbreak in west Africa. Again, the model has been defined in :mod:`.common_models` already. 

.. ipython::

    In [1]: ode = common_models.Legrand_Ebola_SEIHFR()

    In [27]: # initial guess from the paper that studied the outbreak in Congo

    In [28]: theta = numpy.array([0.588,0.794,7.653,     ### the beta
       ....:                      10.0,9.6,5.0,2.0,      ### the omega
       ....:                      7.0,0.81,0.80,         ### alpha, delta, theta
       ....:                      100.,1.0])             ### kappa,intervention time

    In [29]: # initial conditions, note that we have a 0.0 at the end because the model is a non-automonous ode which we have converted the time component out

    In [30]: x0 = numpy.array([1175e4, 0.0, 49.0, 0.0, 0.0, 29.0, 0.0])/population

    In [30]: ode = ode.setInitialValue(x0,t[0]).setParameters(theta)

    In [32]: objLegrand = SquareLoss(theta,ode,x0,t[0],t[1::],y[1::,:],['I','R'],[1175e4,1175e4])

Now, it is important to set additional constraints accurately because a simply box constraint is much larger than the feasible set.  Namely, :math:`\omega_{I},\omega_{D}` are the time taken from onset until end of infectious/death, which has to be bigger than :math:`\omega_{H}`, onset to hospitalization given the nature of the disease.  Therefore, we create extra inequality constraints in addition to the box constraints

.. ipython::

    In [549]: boxBounds = [
       .....:              (0.,100.),  # \beta_I
       .....:              (0.,100.),  # \beta_H
       .....:		   (0.,100.),  # \beta_F 
       .....:		   (0.,100.),  # \oemga_I
       .....:		   (0.,100.),  # \omega_D
       .....:		   (0.,100.),  # \omega_H
       .....:		   (0.,100.),  # \omega_F
       .....:		   (0.,100.),  # \alpha^{-1}
       .....:		   (0.,1.),    # \delta
       .....:		   (0.,1.),    # \theta
       .....:		   (0.,1000.), # \kappa
       .....: 		   (0.,218.)   # intervention tine 
       .....: 		   ]

    In [550]: cons = ({'type': 'ineq', 'fun' : lambda x: numpy.array([x[3]-x[5],x[4]-x[5]]) })

We can now try to find the optimal values, but because this is a difficult problem that takes a very long time without guarantee on the quality of solution

.. ipython:: 

    In [213]: res = scipy.optimize.minimize(fun=objLegrand.cost,
       .....:                               jac=objLegrand.sensitivity,
       .....:                               x0=theta,
       .....:                               bounds=boxBounds,
       .....:                               method='SLSQP')

    In [214]: print(res)

    @savefig ebola_legrand_runtime.png
    In [217]: objLegrand.plot()

    In [217]: plt.close()

As the estimate doesn't appear to provide anything sensible, we also provide a set of values previously obtained here

.. ipython::

    In [1]: theta = numpy.array([  2.28797259e-02,   8.24214614e+01,   7.07217158e+01,
       ...:                     9.95187213e+01,   9.97233885e+01,   9.00143349e+01,
       ...:                     9.47106900e+01,   2.54230760e-02,   2.71633661e-01,
       ...:                     5.07696531e-03,   8.17037312e+02,   3.09602313e+00])

    In [2]: objLegrand.cost(theta)

    In [3]: solution = ode.integrate(t[1::])

    In [4]: f,axarr = plt.subplots(2,3)
    
    In [5]: axarr[0,0].plot(t,solution[:,0]);

    In [5]: axarr[0,0].set_title('Susceptible');

    In [5]: axarr[0,1].plot(t,solution[:,1]);

    In [5]: axarr[0,1].set_title('Exposed');
    
    In [5]: axarr[0,2].plot(t,solution[:,2]);

    In [5]: axarr[0,2].plot(t,y[:,0],'r');
    
    In [5]: axarr[0,2].set_title('Infectious');

    In [5]: axarr[1,0].plot(t,solution[:,3]);

    In [5]: axarr[1,0].set_title('Hospitalised');

    In [5]: axarr[1,1].plot(t,solution[:,4]);

    In [5]: axarr[1,1].set_title('Awaiting Burial');

    In [5]: axarr[1,2].plot(t,solution[:,5]);
    
    In [5]: axarr[1,2].plot(t,y[:,1],'r');

    In [5]: axarr[1,2].set_title('Removed');
    
    In [5]: plt.xlabel('Days from outbreak');
    
    In [5]: plt.ylabel('Population');
    
    In [5]: f.tight_layout();

    @savefig ebola_seihfr_straight_loose.png
    In [5]: plt.show()
    
    In [5]: plt.close()

Evidently, the estimated parameters are very much unrealistic given that a lot of them are near the boundaries.  It is also known from other sources that some of the epidemiology properties of Ebola, with incubation period of around 2 weeks and a mortality rate of around 80 percent.

We can put a slightly stricter bounds on the parameter space, such as 

.. ipython::

    In [549]: boxBounds = [
       .....:              (0.,20.),   # \beta_I
       .....:              (0.,20.),   # \beta_H
       .....:		   (0.,20.),   # \beta_F 
       .....:		   (0.,50.),   # \oemga_I
       .....:		   (0.,50.),   # \omega_D
       .....:		   (0.,50.),   # \omega_H
       .....:		   (0.,50.),   # \omega_F
       .....:		   (0.,30.),   # \alpha^{-1}
       .....:		   (0.,1.),    # \delta
       .....:		   (0.,1.),    # \theta
       .....:		   (0.,1000.), # \kappa
       .....: 		   (0.,218.)   # intervention tine 
       .....: 		   ]

but again, we found that the estimates of the parameters to be virtually on the boundary.  The estimates are repeat here 

.. ipython::

    In [1]: theta = numpy.array([3.96915071e-02,   1.72302620e+01,   1.99749990e+01,
       ...:                      2.67759445e+01,   4.99999990e+01,   5.56122691e+00,
       ...:                      4.99999990e+01,   8.51599523e+00,   9.99999000e-01,
       ...:                      1.00000000e-06,   3.85807562e+00,   1.88385318e+00])

    In [2]: objLegrand.cost(theta)

Note how our objective value is basically now double the previous estimate.  The corresponding figure shown below

.. ipython:: 

    In [3]: solution = ode.integrate(t[1::])

    In [4]: f,axarr = plt.subplots(2,3)
    
    In [5]: axarr[0,0].plot(t,solution[:,0]);

    In [5]: axarr[0,0].set_title('Susceptible');

    In [5]: axarr[0,1].plot(t,solution[:,1]);

    In [5]: axarr[0,1].set_title('Exposed');
    
    In [5]: axarr[0,2].plot(t,solution[:,2]);

    In [5]: axarr[0,2].plot(t,y[:,0],'r');
    
    In [5]: axarr[0,2].set_title('Infectious');

    In [5]: axarr[1,0].plot(t,solution[:,3]);

    In [5]: axarr[1,0].set_title('Hospitalised');

    In [5]: axarr[1,1].plot(t,solution[:,4]);

    In [5]: axarr[1,1].set_title('Awaiting Burial');

    In [5]: axarr[1,2].plot(t,solution[:,5]);
    
    In [5]: axarr[1,2].plot(t,y[:,1],'r');

    In [5]: axarr[1,2].set_title('Removed');
    
    In [5]: plt.xlabel('Days from outbreak');
    
    In [5]: plt.ylabel('Population');

    In [5]: f.tight_layout();
    
    @savefig ebola_seihfr_straight_strict.png
    In [5]: plt.show()

    In [5]: plt.close()


