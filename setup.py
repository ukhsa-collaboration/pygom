#!/bin/env python
"""
@author: Edwin Tye (Edwin.Tye@phe.gov.uk)
"""
import re
import subprocess
from setuptools import setup

with open('LICENSE.txt', 'r') as f:
    license_file = f.read()

with open('README.rst', 'r') as f:
    readme = f.read()

setup_requires=[
    'setuptools-scm>=3.2.0',
    'setuptools_scm_git_archive'
    ]

install_requires = [
    'dask>=0.13.0',
    'matplotlib>=1.0.0',
    'pandas>=0.15.0',
    'python-dateutil>=2.0.0',
    'python-libsbml>=5.0.0',
    'numpy>=1.6.0',
    'scipy>=0.10.0',
    'sympy>=1.0.0'
]

setup(
    name='pygom',
    use_scm_version=True,
    description='ODE modeling in Python',
    long_description=readme,
    license=license_file,
    url='https://github.com/PublicHealthEngland/pygom',
    author="Edwin Tye",
    author_email="Edwin.Tye@phe.gov.uk",
    packages=[
        'pygom',
        'pygom.model',
        'pygom.loss',
        'pygom.utilR'
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_suite='tests',
    scripts=[]
    )
