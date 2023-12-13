#!/usr/bin/env python3
import os
import shutil
from setuptools import Distribution
from setuptools import Extension
from distutils.command.build_ext import build_ext

extra_compile_args = [] if os.name == 'nt' else ["-g", "-O2"]
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
    distribution = Distribution({"name": "cv_algorithm", "ext_modules": ext_modules})
    distribution.package_dir = {"cv_algorithm": "cv_algorithm"}
    
    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()
    
    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)

if __name__ == "__main__":
    build()
