#!/usr/bin/env python3
# =========================================================================================
# (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# =========================================================================================

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="parthenon_tools",
    description="Python helper scripts to be used with Parthenon",
    long_description=long_description,
    url="https://github.com/lanl/parthenon/tree/develop/scripts/python/parthenon_tools",
    author="The Parthenon Team",
    author_email="jonahm@lanl.gov",
    license="LICENSE.txt",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research/DevOps",
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="simulations science computing githubapp",
    packages=find_packages(),
    install_requires=["h5py", "matplotlib", "numpy", "argparse", "cython"],
)
