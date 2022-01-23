#!/usr/bin/env python
import os
import sys
import io
try:
    import setuptools
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
from setuptools import setup, Extension
from setuptools import find_packages

extra_compile_args = [] if os.name == 'nt' else ["-g", "-O2", "-march=native"]
extra_link_args = [] if os.name == 'nt' else ["-g"]

mod_cv_algorithms = Extension('cv_algorithms._cv_algorithms',
                         sources=['src/thinning.cpp',
                                  'src/distance.cpp',
                                  'src/grassfire.cpp',
                                  'src/popcount.cpp',
                                  'src/neighbours.cpp'],
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args)

setup(
    name='cv_algorithms',
    license='Apache license 2.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=['cffi>=1.10'],
    ext_modules=[mod_cv_algorithms],
    test_suite='nose.collector',
    tests_require=['nose', 'coverage', 'mock', 'rednose', 'nose-parameterized'],
    setup_requires=['nose>=1.0'],
    platforms="any",
    zip_safe=False,
    version='1.0.0',
    long_description=io.open("README.rst", encoding="utf-8").read(),
    description='Optimized OpenCV extra algorithms for Python',
    url="https://github.com/ulikoehler/"
)
