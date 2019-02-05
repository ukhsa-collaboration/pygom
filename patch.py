'''
Created on 4 Feb 2019

@author: thomas.finnie
'''
import uuid

from schematics.models import Model
from schematics.types import StringType, UUIDType, IntType
from schematics.types.compound import ModelType, ListType


from . movement import Flux


class Patch(Model):  #pylint: disable=too-few-public-methods
    '''
    Holds the information about a real patch
    '''
    name = StringType()
    patch_id = UUIDType(default=uuid.uuid4)

class PsudoPatch(Model):
    '''
    Holds the information on a psudo-patch

    A psudo-patch is a real-patch at a given time with all the different
    `Flows` as strata
    '''
    psudo_patch_id = UUIDType(default=uuid.uuid4)
    time_point = IntType()
    flux = ListType(ModelType(Flux))
