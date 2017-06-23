"""
    .. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

    Module with functions that deals with epijson and the extraction of data

"""

import json, functools
import pandas as pd
import numpy as np

from dateutil import parser, tz

def epijsonToDataFrame(inData, full_output=False):
    '''
    Obtain a :class:`pandas.DataFrame' given EpiJSON input. The data is
    aggregated (through time).  The original can be obtained by setting
    full_output to True

    Parameters
    ----------
    inData: dict or str
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
    if isinstance(inData, str):
        try:
            epijson = json.loads(inData)
        except:
            with open(inData, 'r') as fp:
                epijson = json.load(fp)
    elif isinstance(inData, bytes):
        epijson = json.loads(inData.decode())
    else:
        epijson = inData

    allRecords = checkEpijsonFormat(epijson)
    
    # obtain the data into (event, date)
    f = lambda x: list(map(lambda x1: (x1['name'], x1['date']), x))
    dataTuple = map(f, allRecords)
    ## dataTuple = map(lambda x: list(map(lambda x1: (x1['name'], x1['date']), x)), allRecords)

    # combining the records as information of the individual is
    # unimportant from pygom point of view
    dataTuple = functools.reduce(lambda x,y: x + y, list(dataTuple))

    # parse the dates under ISO 8601
    data = map(lambda x_y: (x_y[0], _parseDate(x_y[1])), dataTuple)
    # making sure that we have time zone information

    # we put the data info in a dict format to
    # 1. get the unique time stamps
    # 2. speed up indexing / locating events
    dataDict = dict()
    colName = set()
    for name, date in data:
        dataDict.setdefault(date, list())
        dataDict[date] += [str(name)]
        colName.add(str(name))

    colName = list(colName)
    rowName = sorted(dataDict.keys())

    dataList = [_eventNameToVector(dataDict[date], colName) for date in rowName]
    df = pd.DataFrame(dataList, index=rowName, columns=colName)

    if full_output:
        return df, epijson
    else:
        return df.cumsum()

def _eventNameToVector(x, nameList):
    y = np.zeros(len(nameList))
    for name in x:
        y[nameList.index(name)] += 1
    return y

def _parseDate(x):
    y = parser.parse(x)
    if y.tzinfo is None:
        y = parser.parse(x, tzinfos=tz.tzutc)
    return y

def checkEpijsonFormat(epijson, returnRecord=True):
    '''
    Simple checks to see whether the input follows the EpiJSON schema

    Parameters
    ----------
    epijson: dict
        data
    returnRecord: bool, optional
        defaults to True, which outputs the records within the data,
        else the function returns None
    '''
    assert isinstance(epijson, dict), "EpiJSON should be stored as a dict"

    f = map(lambda x: x in ('metadata', 'records'), epijson.keys())
    assert sum(f) == 2, "Does not appear to be a valid EpiJSON format"

    y = epijson['records']
    assert _checkUniqueID(y) == True, "Records id not unique"

    # verify the uniqueness of the id in each record
    # this is relatively unimportant
    y1 = list(map(lambda x: x['events'], y))
    assert sum(map(_checkUniqueID, y1)) == len(y1), \
        "Events id not unique in records"

    return y1 if returnRecord else True
    
def _checkUniqueID(y):
    '''
    Check if the input y is a set or a bag.  Returns
    True if it is a set, False otherwise
    '''
    ids = list(map(lambda x: x['id'], y))    
    return len(ids) == len(set(ids))

