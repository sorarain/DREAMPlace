from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

import Cython.Compiler.Options
Cython.Compiler.Options.annotate=True

setup(ext_modules = cythonize(Extension(
    "net_span",['net_span.pyx'],
    extra_compile_args=["-O3","-fopenmp"],
    extra_link_args=['-fopenmp']
)))