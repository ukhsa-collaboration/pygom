"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module with functions that deals with epijson and the extraction of data

"""

import functools
import json

from dateutil import parser, tz

import pandas as pd
import numpy as np

def epijson_to_data_frame(in_data, full_output=False):
    '''
    Obtain a :class:`pandas.DataFrame' given EpiJSON input. The data is
    aggregated (through time).  The original can be obtained by setting
    full_output to True

    Parameters
    ----------
    in_data: dict or str
        dict that conforms to the EpiJSON format or a str
        which will be read as JSON
    full_output: bool, optional
        defaults to False, if True, returns
        (:class:`pandas.DataFrame`, dict) where the first is element is not in
        the aggregated form and the second is the JSON data.  Invoke
        df.cumsum() to get the same output when full_output=False

    Returns:
    df: :class:`pandas.DataFrame`
        data in as cumulative sum where the row represent the
        unique time stamps and column the events
    '''
    if isinstance(in_data, str):
        try:
            epijson = json.loads(in_data)
        except:
            with open(in_data, 'r') as fp:
                epijson = json.load(fp)
    elif isinstance(in_data, bytes):
        epijson = json.loads(in_data.decode())
    else:
        epijson = in_data

    allRecords = check_epijson_format(epijson)

    # obtain the data into (event, date)
    f = lambda x: list(map(lambda x1: (x1['name'], x1['date']), x))
    data_tuple = map(f, allRecords)
    ## dataTuple = map(lambda x: list(map(lambda x1: (x1['name'], x1['date']), x)), allRecords)

    # combining the records as information of the individual is
    # unimportant from pygom point of view
    data_tuple = functools.reduce(lambda x,y: x + y, list(data_tuple))

    # parse the dates under ISO 8601
    data = map(lambda x_y: (x_y[0], _parseDate(x_y[1])), data_tuple)
    # making sure that we have time zone information

    # we put the data info in a dict format to
    # 1. get the unique time stamps
    # 2. speed up indexing / locating events
    data_dict = dict()
    col_name = set()
    for name, date in data:
        data_dict.setdefault(date, list())
        data_dict[date] += [str(name)]
        col_name.add(str(name))

    col_name = list(col_name)
    row_name = sorted(data_dict.keys())

    data_list = [_eventNameToVector(data_dict[d], col_name) for d in row_name]
    df = pd.DataFrame(data_list, index=row_name, columns=col_name)

    if full_output:
        return df, epijson
    else:
        return df.cumsum()

def _eventNameToVector(x, name_list):
    y = np.zeros(len(name_list))
    for name in x:
        y[name_list.index(name)] += 1
    return y

def _parseDate(x):
    y = parser.parse(x)
    if y.tzinfo is None:
        y = parser.parse(x, tzinfos=tz.tzutc)
    return y

def check_epijson_format(epijson, return_record=True):
    '''
    Simple checks to see whether the input follows the EpiJSON schema

    Parameters
    ----------
    epijson: dict
        data
    return_record: bool, optional
        defaults to True, which outputs the records within the data,
        else the function returns None
    '''
    assert isinstance(epijson, dict), "EpiJSON should be stored as a dict"

    f = map(lambda x: x in ('metadata', 'records'), epijson.keys())
    assert sum(f) == 2, "Does not appear to be a valid EpiJSON format"

    y = epijson['records']
    assert _checkUniqueID(y) is True, "Records id not unique"

    # verify the uniqueness of the id in each record
    # this is relatively unimportant
    y1 = list(map(lambda x: x['events'], y))
    assert sum(map(_checkUniqueID, y1)) == len(y1), \
        "Events id not unique in records"

    return y1 if return_record else True

def _checkUniqueID(y):
    '''
    Check if the input y is a set or a bag.  Returns
    True if it is a set, False otherwise
    '''
    ids = list(map(lambda x: x['id'], y))
    return len(ids) == len(set(ids))
