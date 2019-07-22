'''
Created on 22 Jul 2019

@author: Thomas.Finnie
'''

from schematics.models import Model
from schematics.types import FloatType
from schematics.types.compound import ModelType, ListType, DictType

from .metadata import MetaData



class ResultsByName(Model):
    '''
    The results of a model run with patches identified by name
    '''
    metadata = ModelType(MetaData, default=MetaData(), required=True)
    results = DictType(ListType(DictType(FloatType)),
                       required=True,
                       default={})
