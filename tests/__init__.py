# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:47:65 2018

@author: Edwin Tye
"""

import unittest

def test_suite_loader():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite
