'''
Created on 19 Jul 2019

@author: Thomas.Finnie
'''
import os
import datetime
import socket
import uuid

from schematics.models import Model
from schematics.types import StringType, UUIDType, IntType
from schematics.types.compound import ModelType, ListType

from quilty import __version__
from .model import ParameterListType

class MetaData(Model):
    '''
    Holds the metadata on a run
    '''
    # A succinct description of what is in the file
    title = StringType(default='Quilty patch modelling framework output')
    # The short name of the model. This should be URL friendly. This may be used
    # in other systems to provide web access to results.
    model_name = StringType(default='quilty')
    # Ideally this is descriptive of what the model is and does.
    model_long_name  = StringType(default='Quilty modelling framework')
    # The version of the model.
    model_version = StringType(default=__version__)
    # Free text to fill the source parameter but may change to being/requiring
    # DIds (Data Ids) for referencing.
    data_source = StringType()
    # Where was this file was produced?
    institution = StringType()
    # This attribute provides an audit trail for the file (like NetCDF). By
    # default a timestamp followed by the model name, model version, username
    # and machine.
    history = StringType(default='')
    # Universally Unique IDentifier for this model run. Must follow
    # the Open Software Foundation UUID standard.
    run_uuid = UUIDType(default=uuid.uuid4)
    # The parameters that the model was run with.
    parameters = ParameterListType()
    # Published or web based references that describe the data or the methods
    # used to produce it.
    references = StringType()
    #  Free text for additional comments.
    comment = StringType()

    def __init__(self, **kwargs):
        '''
        Small overload to automatically capture history if not provided
        '''
        super().__init__(**kwargs)
        if self.history == '':
            self.history = '{} {} {} {} {}'.format(datetime.datetime.utcnow().isoformat(),
                                                   self.model_name,
                                                   self.model_version,
                                                   os.getlogin(),
                                                   socket.gethostname()
                                                   )
