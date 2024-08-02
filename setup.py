#!/bin/env python
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy

# Options for the compiled modules
Options.docstrings = True
Options.annotate = False

extensions = [
    Extension("pygom.model._tau_leap",
              ["src/pygom/model/_tau_leap.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-std=c99'],
#                  extra_compile_args=['-fopenmp'],
#                  extra_link_args=['-fopenmp']),
             )
]

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": 3, "profile": False}))
