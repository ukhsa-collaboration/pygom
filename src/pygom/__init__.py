''' pygom

.. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

'''
from importlib.metadata import PackageNotFoundError, version

from .loss import *
from .model import *
#from .utilR import *

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
