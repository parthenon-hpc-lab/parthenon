.. _tutorial

Writing your first Parthenon-based Code
=========================================

In this tutorial, we will walk through how to write a Parthenon-based
code from scratch. We'll build a simple toy code that rotates an
ellipse in a circle, with AMR, to demonstrate the elements that make
up a Parthenon code and high-level Parthenon concepts.

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

Directory structure
---------------------

The most common way to include Parthenon in a project is to build it
*in-tree*. This means creating a repository for your code and
including Parthenon *inside* it. This typically looks like:

.. code-block::

   ellipse/
   ├── CMakeLists.txt
   ├── external
   │   └── parthenon
   └── src

where here I've assumed we named our code *ellipse*. The source code
for the new ellipse executable will live in ``src``, and Parthenon
will live in ``external/parthenon``. Note the ``CMakeLists.txt`` file,
we'll come back to that.

The most common way to include Parthenon in a project under git
version control is ``git-submodules``, which allow a git repository to
be included inside another git repository such that the source code
for the dependency isn't directly committed into the downstream
project. Lets set it up. You can get to the project structure with:

.. code-block:: bash

   mkdir ellipse
   cd ellipse
   git init
   mkdir external
   mkdir source
   touch CMakeLists.txt
   git submodule add git@github.com:parthenon-hpc-lab/parthenon.git external/parthenon

Parthenon itself also has submodules. We need to clone them for a
Parthenon-based project to build. Do so via

.. code-block:: bash

   git submodule update --init --recursive

You can now commit files and push as you normally would. If you want
to update parthenon, simply go inside the parthenon directory inside
your project, checkout the relevant release or branch and pull. Then
you can commit the folder as if you were working with raw source code
and git will do the right thing.

.. note::

   Parthenon also has a ``spackage``. You can see details in our
   :ref:`build doc <building>`.

The top-level ``CMakeLists.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cmake`` is a configuration language. It tells your computer how find
and tie together dependencies and builds a ``makefile`` which actually
calls the compiler to build your code. The top-level
``CMakeLists.txt`` file contains some of these details. Open the file
and edit to look like this:

.. code-block:: cmake

   # Names the project ellipse
   project(ellipse LANGUAGES C CXX)
   # We require C++20
   set(CMAKE_CXX_STANDARD 20)
   # A useful command for debugging
   set(CMAKE_EXPORT_COMPILE_COMMANDS On)

   # This is just a safety thing, but I recommend it including it. It
   # forces you to build the code in a directory that isn't the same as
   # your source code.
   file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
   if(EXISTS "${LOC_PATH}")
     message(FATAL_ERROR
      "You cannot build in a source directory (or any directory with a CMakeLists.txt file). "
      "Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
   endif()

   # Mostly a convenience thing. If you don't specify which flags to
   # compile with, cmake prefers a recipe "RelWithDebInfo" which is a
   # mix of code speed and debugging. For maximum performance, specify
   # -DCMAKE_BUILD_TYPE=Release. For debugging, specify
   # -DCMAKE_BUILD_TYPE=Debug.
   set(default_build_type "RelWithDebInfo")
   if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
     message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
     set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
       STRING "Choose the type of build." FORCE)
     # Set the possible values of build type for cmake-gui
     set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
       "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
   endif()

   # Parthenon can also be built standalone. But since we're building
   # it as part of our ellipse project, let's disable the tests and
   # example code that come with it.
   set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "" FORCE)
   set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
   # add Parthenon
   add_subdirectory(external/parthenon parthenon)

   # This command will error out currently, but we want it to tell
   # cmake to look for our source code once we write some
   add_subdirectory(src)

A fully-featured project may have many more things in the top-level
cmake, such as code for unit tests and additional dependency
handling. But we'll stick with this for now.

Now lets start writing some code and discussing some high-level
Parthenon concepts.

High-level Parthenon concepts
-------------------------------

A Parthenon-based project consists of:

* Any number of *packages*, which, conceptually, own work to do and state to do it on.

* A *driver* which orchestrates work.

* At least one *problem generator* which provides initial conditions for the solver.

* A main function which calls a ``ParthenonManager`` to provide setup/teardown and entry into a program.

Let's go through them each.

Packages
^^^^^^^^^^^


