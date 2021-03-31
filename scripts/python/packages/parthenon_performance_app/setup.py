#!/usr/bin/env python3
#=========================================================================================
# (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#=========================================================================================

from setuptools import setup,find_packages

exec(open('parthenon_performance_app/version.py','r').read())
with open('README.md','r') as f:
      long_description = f.read()

setup(
    name='parthenon_performance_app',
    version=__version__,
    description='Parthenon metrics application for checking for performance regressions',
    long_description=long_description,
    url='https://github.com/lanl/parthenon/tree/develop/scripts/python/packages/parthenon_performance_app',
    author = 'Joshua S. Brown',
    author_email = 'joshbro42867@yahoo.com',
    license_files = 'LICENSE.txt',
    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: DevOps',
      'Environment :: Console',
      'Natural Language :: English',
      'Operating System :: Unix',
      'Programming Language :: Python :: 3.9',
      ],
    keywords = 'testing performance regression',
    packages = find_packages(),
    install_requires = ['matplotlib','numpy','pyjwt','argparse','pycurl','pem','gitpython'],
    scripts=['bin/parthenon_metrics_app.py']
    )
