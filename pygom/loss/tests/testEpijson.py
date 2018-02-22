from unittest import TestCase

import pkgutil

from pygom import common_models
from pygom.loss.read_epijson import epijsonToDataFrame
from pygom.loss.epijson_loss import EpijsonLoss

import numpy as np

class TestEpijson(TestCase):

    def test_read_epijson(self):
        data = pkgutil.get_data('pygom', 'data/eg1.json')
        df = epijsonToDataFrame(data)
        y = df.values.ravel()
        self.assertTrue(np.all(y == np.array([1.,2.,3.,4.,5.,6.])))

    def test_initialize_epijson_loss(self):
        data = pkgutil.get_data('pygom', 'data/eg1.json')
        ode = common_models.SIR()
        ode.parameters = [0.5, 0.3]
        obj = EpijsonLoss([0.005, 0.03], ode, data, 'Death', 'R', [300, 2, 0])
        self.assertTrue(np.allclose(obj.cost(), 10.86559460256))
