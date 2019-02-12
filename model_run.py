'''
Created on 12 Feb 2019

@author: Thomas.Finnie
'''

from schematics.models import Model
from schematics.types.compound import ModelType, ListType
from schematics.types import (StringType,
                              FloatType,
                              UUIDType,
                              ModelType,
                              IntType
                              )

class PsudoPatchRun(Model):
    '''
    The run parameters for a psudo-patch

    Parameters
    ----------
    start_time: The time the whole model starts at
    end_time: The time the whole model stops at
    stride: The time period a psudo_patch is occupied. (NB. time periods within
      a circad do not have to be even)
    skip: The period of time a psudo-patch should skip between the end of one
      period and the beginning of the next
    '''
    start_time = FloatType(requited=True)
    end_time = FloatType(required=True)
    stride = FloatType(required=True)
    skip = FloatType(required=True)
