#!/usr/bin/env python3
import os
from setuptools import Distribution
from setuptools import Extension
from distutils.command.build_ext import build_ext

extra_compile_args = [] if os.name == 'nt' else ["-g", "-O2", "-march=ivybridge"]
extra_link_args = [] if os.name == 'nt' else ["-g"]
platform_src = ["src/windows.cpp"] if os.name == 'nt' else []

ext_modules = [
    Extension('cv_algorithms._cv_algorithms',
        include_dirs = [os.path.join(os.path.dirname(__file__), "src")],
        sources=['src/thinning.cpp',
                'src/distance.cpp',
                'src/grassfire.cpp',
                'src/popcount.cpp',
                'src/neighbours.cpp'] + platform_src,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)
]

class BuildFailed(Exception):
    pass

class ExtBuilder(build_ext):

    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed('File not found. Could not compile C extension.')

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed('Could not compile C extension.')


def build():
    """
    This function is mandatory in order to build the extensions.
    """
    distribution = Distribution({"name": "extended", "ext_modules": ext_modules})
    distribution.package_dir = {"extended": "extended"}
    
    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    #setup_kwargs.update(
    #    {
    #        "ext_modules": ext_modules,
    #        "cmdclass": {"build_ext": ExtBuilder},
    #        "zip_safe": False,
    #    }
    #)
    #print(setup_kwargs)
    
if __name__ == "__main__":
    build()
