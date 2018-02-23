.. _epijson:

******************************
Reading and using EpiJSON data
******************************

Epidemiology data is complicated due to the many different stages a patient can go through and whether a modeling technique is applicable depends heavily on the recording of data.  EpiJSON is a framework whih tries to captures all the information [Finnie2016]_, in a JSON format as the name suggests.  

This package provides the functionality to process EpiJSON data.  Due to the nature of this package, modeling of ode, it processes the data file with this in mind.  The output is therefore in the cumulative form as default, shown below, in a :class:`pandas.DataFrame` format.  The input can be in a string format, a file or already a :class:`dict`.

.. ipython::

    In [1]: from pygom.loss.read_epijson import epijson_to_data_frame

    In [2]: import pkgutil

    In [3]: data = pkgutil.get_data('pygom', 'data/eg1.json')

    In [3]: df = epijson_to_data_frame(data)

    In [4]: print(df)

Given that the aim of loading the data is usually for model fitting, we allow EpiJSON as input directly to the loss class :class:`pygom.loss.EpijsonLoss` which uses the Poisson loss under the hood.  

.. ipython::

    In [1]: from pygom.model import common_models

    In [2]: from pygom.loss.epijson_loss import EpijsonLoss

    In [3]: ode = common_models.SIR([0.5, 0.3])

    In [4]: obj = EpijsonLoss([0.005, 0.03], ode, data, 'Death', 'R', [300, 2, 0])

    In [5]: print(obj.cost())

    In [6]: print(obj._df)

Given an initialized object, all the operations are inherited from :class:`pygom.loss.BaseLoss`.  We demonstrated above how to calculate the cost and the rest will not be shown for brevity.  The data frame is stored inside of the loss object and can be retrieved for inspection at any time point.

Rather unfortunately, initial values for the states is still required, but the time is not.  When the time is not supplied, then the first time point in the data will be treated as :math:`t0`.  The input `Death` indicate which column of the data is used and :math:`R` the corresponding state the data belongs to.

