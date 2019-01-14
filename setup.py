#!/bin/env python
'''
@author: Edwin Tye (Edwin.Tye@phe.gov.uk)
'''
from setuptools import setup

with open('LICENSE.txt') as f:
    license_file = f.read()

with open('README.rst') as f:
    readme = f.read()

cmdclass = {}
ext_modules =[]
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

setup(name='pygom',
      version='0.1.3',
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
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      test_suite='tests.test_suite_loader',
      scripts=[]
)
