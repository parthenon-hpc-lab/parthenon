.. _tutorial

Writing your first Parthenon-based Code
=========================================

In this tutorial, we will walk through how to write a Parthenon-based
code from scratch. We'll build a simple toy code that rotates an
ellipse in a circle, with AMR, to demonstrate the elements that make
up a Parthenon code and high-level parthenon concepts.

Prerequisites
---------------

Parthenon requires at a minimum a C++20 compiler, git, and cmake. Most
real applications also require an MPI library (MPI stands for message
passing interface) for parallelism. In this tutorial, we'll also be
relying on HDF5 for output and numpy, matplotlib, and h5py for
visualization. On Ubuntu Linux, you can install the non-python
dependencies as

.. code-block:: bash

   sudo apt install build-essential libmpich-dev libhdf5-mpich-dev hdf5-tools git cmake

For Python, use your preferred python environment. I suggest a
project-specific Python virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install numpy matplotlib h5py

.. note::

   Python and cmake can interfere with each other. I find this is
   especially true with Anaconda and friends, as Anaconda can install,
   e.g., a serial version of hdf5, which cmake finds when it
   configures. Thus, I recommend activating your virtual environment
   but leaving your conda environment inactive.

