"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    This extends PoissonLoss to take care of case where EpiJSON
    data is used rather than providing observations in as a series
    of tuple (y,t)

"""
__all__ = ['EpijsonLoss']

import datetime
from dateutil import parser, tz

import pandas as pd
import numpy as np

from .ode_loss import PoissonLoss
from .read_epijson import epijson_to_data_frame

secondsInDay = float(24 * 60 * 60)

class EpijsonLoss(PoissonLoss):
    '''
    Uses EpiJSON as input and assumes that we are taking Poisson Loss.
    The initial condition x0 has
    '''
    def __init__(self, theta, ode, epijson, col_name, state_name, x0,
                 t0=None, target_param=None, target_state=None):

        dfNonAggre, self._epijson = epijson_to_data_frame(epijson, True)
        df = dfNonAggre.cumsum()

        if isinstance(col_name, str):
            col_name = [col_name]
        elif not hasattr(col_name, '__iter__'):
            raise Exception("col_name should be a string or iterable")

        assert sum(map(lambda x: x in df.columns, col_name)) == len(col_name), \
            "Not all column name can be found, columns in data are: %s" % \
            df.comlumns

        if t0 is None:
            t0 = df.index[0]
        else:
            if isinstance(t0, str):
                t0 = parser.parse(t0)
                # we should always put things as utc if possible and
                # convert it if that is not the case
                if t0.tzinfo is None:
                    t0 = pd.Timestamp(t0, tz=tz.tzutc())
                else:
                    t0 = pd.Timestamp(t0).tz_convert(tz.tzutc())
            elif isinstance(t0, (float, int)):
                t0 = df.index[0] + datetime.timedelta(days=t0)

        # generate the time difference and if there are any negative
        # we just take the positives.  This also defines the starting
        # time point at zero
        tau = df.index - t0
        tau = np.array(tau.total_seconds()/secondsInDay)

        X = df.loc[:,col_name].values[tau > 0]
        T = tau[tau > 0]

        self._df = df

        super(EpijsonLoss, self).__init__(theta, ode, x0, 0, T, X,
                                          state_name,
                                          target_param, target_state)

    def __repr__(self):
        return "EpijsonLoss"+self._get_model_str()

    @property
    def data_frame(self):
        '''
        Returns the a :class:`pandas.DataFrame` that represents the
        EpiJSON data
        '''
        return self._df