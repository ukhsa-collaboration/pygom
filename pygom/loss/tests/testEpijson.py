from unittest import TestCase

import pkgutil

from pygom import common_models
from pygom.loss.read_epijson import epijsonToDataFrame
from pygom.loss.epijson_loss import EpijsonLoss

import numpy

class TestEpijson(TestCase):

    def test_read_epijson(self):
        data = pkgutil.get_data('pygom', 'data/eg1.json')
        df = epijsonToDataFrame(data)
        y = df.values.ravel()
        assert numpy.all(y == numpy.array([1.,2.,3.,4.,5.,6.])), \
        "Error reading EpiJSON data"

    def test_initialize_epijson_loss(self):
        data = pkgutil.get_data('pygom', 'data/eg1.json')
        ode = common_models.SIR()
        ode.parameters = [0.5, 0.3]
        obj = EpijsonLoss([0.005, 0.03], ode, data, 'Death', 'R', [300, 2, 0])
        assert abs(obj.cost() - 10.86559460256) <= 0.001, "Error with loss obj"
