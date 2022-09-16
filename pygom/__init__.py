# -*- coding: utf-8 -*-
# @Author: Martin Grunnill 
# @Date:   2020-07-03 07:28:58
# @Last Modified by:   Martin Grunnill 
# @Last Modified time: 2022-01-18 16:42:11
''' pygom

.. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

'''
from pkg_resources import get_distribution, DistributionNotFound

from .loss import *
from .model import *
#from .utilR import *

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
