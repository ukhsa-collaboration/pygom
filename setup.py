#!/bin/env python
'''
@author: Edwin Tye (Edwin.Tye@phe.gov.uk)
'''
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

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

tests_require = [
    'nose>=1.1'
]

setup(name='pygom',
      version='0.1.1',
      description='ODE modeling in Python',
      long_description=readme(),      
      url='https://github.com/PublicHealthEngland/pygom',
      author="Edwin Tye",
      author_email="Edwin.Tye@phe.gov.uk",
      packages=[
          'pygom',
          'pygom.model',
          'pygom.loss',
          'pygom.utilR'
      ],
      license='LICENCE.txt',
      install_requires=install_requires,
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      tests_require=tests_require,
      test_suite='nose.collector',
      scripts=[]
)
