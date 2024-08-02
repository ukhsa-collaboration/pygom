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
