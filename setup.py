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
    'enum34>=1.0.4',
    'graphviz',
    'ipython',
    'matplotlib>=1.0.0',
    'numpy>=1.6.0',
    'scipy>=0.10.0',
    'sympy>=0.7.0'
]

tests_require = [
    'nose>=1.1'
]

setup(name='pygom',
      version='0.1.0',
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
