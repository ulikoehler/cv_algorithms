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
platform_src = ["src/windows.cpp"] if os.name == 'nt' else []

mod_cv_algorithms = Extension('cv_algorithms._cv_algorithms',
                         include_dirs = [os.path.join(os.path.dirname(__file__), "src")],
                         sources=['src/thinning.cpp',
                                  'src/distance.cpp',
                                  'src/grassfire.cpp',
                                  'src/popcount.cpp',
                                  'src/neighbours.cpp'] + platform_src,
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args)

setup(
    name='cv_algorithms',
    license='Apache license 2.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=['cffi>=0.7'],
    ext_modules=[mod_cv_algorithms],
    test_suite='nose.collector',
    tests_require=['nose', 'coverage', 'mock', 'rednose', 'nose-parameterized'],
    setup_requires=['nose>=1.0'],
    platforms="any",
    zip_safe=False,
    version='1.0.4',
    description='Optimized OpenCV extra algorithms for Python',
    url="https://github.com/ulikoehler/cv_algorithms"
)
