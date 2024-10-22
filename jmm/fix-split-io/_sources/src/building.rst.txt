.. _building:

Building Parthenon
==================

*IMPORTANT: We try our best to keep the instructions up-to-date.
However, Parthenon itself, dependencies, and environments constantly
changes so that the instruction may not work any more. If you come
across a disfunctional setup, please report it by open an issue or
propose an updated description in a pull request*

General list of cmake options:
------------------------------

+-------------------------------------------+--------------------------------+---------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                                    | Default                        | Type    | Description                                                                                                                                                  |
+===========================================+================================+=========+==============================================================================================================================================================+
|| PARTHENON\_SINGLE\_PRECISION             || OFF                           || Option || Enable single precision mode if requested                                                                                                                   |
|| PARTHENON\_DISABLE\_HDF5                 || OFF                           || Option || HDF5 is enabled by default if found, set this to True to disable HDF5                                                                                       |
|| PARTHENON\_DISABLE_HDF5\_COMPRESSION     || OFF                           || Option || HDF5 compression is enabled by default, set this to True to disable compression in HDF5 output/restart files                                                |
|| PARTHENON\_ENABLE\_ASCENT                || OFF                           || Option || Enable Ascent for in situ visualization and analysis                                                                                                        |
|| PARTHENON\_DISABLE\_MPI                  || OFF                           || Option || MPI is enabled by default if found, set this to True to disable MPI                                                                                         |
|| PARTHENON\_ENABLE\_HOST\_COMM\_BUFFERS   || OFF                           || Option || MPI communication buffers are by default allocated on the execution device. This options forces allocation in memory accessible directly by the host.       |
|| PARTHENON\_DISABLE\_SPARSE               || OFF                           || Option || Disable sparse allocation of sparse variables, i.e., sparse variable still work but are always allocated. See also :ref:`sparse doc <sparse compile-time>`. |
|| ENABLE\_COMPILER\_WARNINGS               || OFF                           || Option || Enable compiler warnings                                                                                                                                    |
|| TEST\_ERROR\_CHECKING                    || OFF                           || Option || Enables the error checking unit test. This test will FAIL                                                                                                   |
|| TEST\_INTEL\_OPTIMIZATION                || OFF                           || Option || Test intel optimization and vectorization                                                                                                                   |
|| CHECK\_REGISTRY\_PRESSURE                || OFF                           || Option || Check the registry pressure for Kokkos CUDA kernels                                                                                                         |
|| BUILD\_TESTING                           || ON                            || Option || Enable test (set by CTest itself)                                                                                                                           |
|| PARTHENON\_DISABLE\_EXAMPLES             || OFF                           || Option || Toggle building of examples, if regression tests are on, drivers needed by the tests will still be built                                                    |
|| PARTHENON\_ENABLE\_TESTING               || ${BUILD\_TESTING}             || Option || Default value to enable Parthenon tests                                                                                                                     |
|| PARTHENON\_ENABLE\_INTEGRATION\_TESTS    || ${PARTHENON\_ENABLE\_TESTING} || Option || Enable integration tests                                                                                                                                    |
|| PARTHENON\_ENABLE\_REGRESSION\_TESTS     || ${PARTHENON\_ENABLE\_TESTING} || Option || Enable regression tests                                                                                                                                     |
|| PARTHENON\_ENABLE\_UNIT\_TESTS           || ${PARTHENON\_ENABLE\_TESTING} || Option || Enable unit tests                                                                                                                                           |
|| PARTHENON\_PERFORMANCE\_TESTS            || ${PARTHENON\_ENABLE\_TESTING} || Option || Enable performance tests                                                                                                                                    |
|| NUM\_MPI\_PROC\_TESTING                  || 4                             || String || Number of MPI ranks used for MPI-enabled regression tests                                                                                                   |
|| NUM\_GPU\_DEVICES\_PER\_NODE             || 1                             || String || Number of GPUs per node to use if built with `Kokkos_ENABLE_CUDA`                                                                                           |
|| PARTHENON\_ENABLE\_PYTHON\_MODULE\_CHECK || ${PARTHENON\_ENABLE\_TESTING} || Option || Enable checking if python modules used in regression tests are available                                                                                    |
|| PARTHENON\_ENABLE\_GPU\_MPI\_CHECKS      || ON                            || Option || Enable pre-test gpu-mpi checks                                                                                                                              |
|| REGRESSION\_GOLD\_STANDARD\_VER          || #                             || Int    || Version of current gold standard file used in regression tests. Default is set to latest version matching the source.                                       |
|| REGRESSION\_GOLD\_STANDARD\_HASH         || SHA512=...                    || String || Hash value of gold standard file to be downloaded. Used to ensure that the download is not corrupted.                                                       |
|| REGRESSION\_GOLD\_STANDARD\_SYNC         || ON                            || Option || Create `gold_standard` target to download gold standard files                                                                                               |
|| CODE\_COVERAGE                           || OFF                           || Option || Builds with code coverage flags                                                                                                                             |
|| PARTHENON\_LINT\_DEFAULT                 || OFF                           || Option || Lint the code as part of the default target (otherwise use the `lint` target)                                                                               |
|| PARTHENON\_COPYRIGHT\_CHECK\_DEFAULT     || OFF                           || Option || Check copyright as part of the default target (otherwise use the `check-copyright` target)                                                                  |
|| CMAKE\_INSTALL\_PREFIX                   || machine specific              || String || Optional path for library installation                                                                                                                      |
|| Kokkos\_ROOT                             || unset                         || String || Path to a Kokkos source directory (containing CMakeLists.txt)                                                                                               |
|| PARTHENON\_IMPORT\_KOKKOS                || ON/OFF                        || Option || If ON, attempt to link to an external Kokkos library. If OFF, build Kokkos from source and package with Parthenon                                           |
|| BUILD\_SHARED\_LIBS                      || OFF                           || Option || If installing Parthenon, whether to build as shared rather than static                                                                                      |
+-------------------------------------------+--------------------------------+---------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+


.. note::
   CMake options prefixed with *PARTHENON\_* modify behavior.

.. note::
  **On MPI usage:** By default communication buffers are allocated in the execution device’s
  memory, e.g., directly on the GPU when using Cuda. This requires the MPI
  library to be compiled with support for directly accessing device memory
  (e.g., often referred to as “Cuda-aware MPI”). To force buffer
  allocation in host memory (currently *not* recommended as it typically
  results in a performance degradation) set
  ``PARTHENON_ENABLE_HOST_COMM_BUFFERS=ON``.

Using Parthenon as a Subdirectory
---------------------------------

For simple applications, Parthenon can be added as a subdirectory to
your project. For example, you can add parthenon as a git submodule:

::

   git submodule add https://github.com/parthenon-hpc-lab/parthenon.git

And then you can use parthenon in your CMake project by adding it as a
subdirectory:

.. code:: cmake

   add_subdirectory(path/to/parthenon)

   add_executable(myapp ...)
   target_link_libraries(myapp PRIVATE Parthenon::parthenon)

Installing Parthenon
--------------------

An alternative to building Parthenon as a subdirectory is to first build
Parthenon separately as a library and then link to it when building the
app. Parthenon can be built as either a static (default) or a shared
library.

To build Parthenon as a library, provide a ``CMAKE_INSTALL_PREFIX`` path
to the desired install location to the Parthenon cmake call. To build a
shared rather than a static library, also set ``BUILD_SHARED_LIBS=ON``.
Then build and install (note that ``--build`` and ``--install`` require
CMake 3.15 or greater).

Building as a static library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cmake -DCMAKE_INSTALL_PREFIX="$your_install_dir" $parthenon_source_dir
   cmake --build . --parallel
   cmake --install .

Building as a shared library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cmake -DCMAKE_INSTALL_PREFIX="$your_install_dir" -DBUILD_SHARED_LIBS=ON $parthenon_source_dir
   cmake --build . --parallel
   cmake --install .

When building Parthenon, Kokkos will also be built from source if it
exists in ``parthenon/external`` or at a provided ``Kokkos_ROOT`` by
default. If installing Parthenon, this will also install Kokkos in the
same directory. If ``PARTHENON_IMPORT_KOKKOS=ON`` is provided or no
Kokkos/CMakeLists.txt is found, the build system will attempt to find a
Kokkos installation in the current PATH.

A cmake target, ``lib*/cmake/parthenon/parthenonConfig.cmake`` is
created during installation. To link to parthenon, one can either
specify the include files and libraries directly or call
``find_package(parthenon)`` from cmake.

Linking an app with *make*
~~~~~~~~~~~~~~~~~~~~~~~~~~

The below example makefile can be used to compile the *calculate_pi*
example by linking to a prior library installation of Parthenon. Note
that library flags must be appropriate for the Parthenon installation;
it is not enough to simply provide *-lparthenon*.

.. code:: bash

   PARTHENON_INSTALL=/path/to/your/parthenon/install
   KOKKOS_INSTALL=/path/to/your/Kokkos/install
   CC=g++
   CCFLAGS = -g -std=c++14 -L${PARTHENON_INSTALL}/lib \
    -I${PARTHENON_INSTALL}/include/ \
    -I${KOKKOS_INSTALL}/include/ -L${KOKKOS_INSTALL}/lib
   LIB_FLAGS = -Wl,-rpath,${PARTHENON_INSTALL}/lib -lparthenon \
    -Wl,-rpath,${KOKKOS_INSTALL}/lib -lmpi -lkokkoscore -lhdf5 -ldl \
    -lkokkoscontainers -lz -lpthread -lgomp -lmpi_cxx
   CC_COMPILE = $(CC) $(CCFLAGS) -c
   CC_LOAD = $(CC) $(CCFLAGS)
   .cpp.o:
     $(CC_COMPILE) $*.cpp
   EXE = pi_example
   all: $(EXE)
   SRC = calculate_pi.cpp pi_driver.cpp
   OBJ = calculate_pi.o pi_driver.o
   INC = calculate_pi.hpp pi_driver.hpp
   $(OBJ): $(INC) makefile
   $(EXE): $(OBJ) $(INC) makefile
     $(CC_LOAD) $(OBJ) $(LIB_FLAGS) -o $(EXE)
   clean:
     $(RM) $(OBJ) $(EXE)

Linking an app with *cmake*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The below example ``CMakeLists.txt`` can be used to compile the
*calculate_pi* example with a separate Parthenon installation through
*cmake*\ ’s ``find_package()`` routine.

.. code:: cmake

   cmake_minimum_required(VERSION 3.11)

   project(parthenon_linking_example)
   set(CMAKE_CXX_STANDARD 17)
   set(CMAKE_CXX_EXTENSIONS OFF)
   find_package(parthenon REQUIRED PATHS "/path/to/parthenon/install")
   add_executable(
     pi-example
     pi_driver.cpp
     pi_driver.hpp
     calculate_pi.cpp
     calculate_pi.hpp
     )
   target_link_libraries(pi-example PRIVATE Parthenon::parthenon)

System specific instructions
----------------------------

Common first step: Obtain the Parthenon source including external
dependencies (mostly Kokkos)

.. code:: bash

   # Clone parthenon, with submodules
   git clone --recursive https://github.com/parthenon-hpc-lab/parthenon.git
   export PARTHENON_ROOT=$(pwd)/parthenon

We set the latter variable for easier reference in out-of-source builds.

Default machine configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To make the default configuration on widely used systems easier,
Parthenon provides machine configuration files that contain default
options. Defaults options include, but are not limited to setting - the
compiler (e.g., ``nvcc_wrapper`` for Cuda builds), or - paths to non
default package locations (e.g., for a custom HDF5 install), or - custom
MPI related commands used in the Parthenon test suite (e.g., the launch
command).

The machine configurations shipped with Parthenon are located in
```PARTHENON_ROOT/cmake/machinecfg`` <../cmake/machinecfg>`__ and are
named by the machine name. In order to use them either - set the
``MACHINE_CFG`` environment variable to the appropriate file, or - set
the ``MACHINE_CFG`` CMake variable to the appropriate file. In addition,
you can set the ``MACHINE_VARIANT`` CMake variable to pick a specific
configuration, e.g., one with Cuda and MPI enabled.

We suggest to inspect the corresponding file for available options on a
specific machine.

In general, a typical workflow is expected to create your own machine
file, e.g., on your develop system. We suggest to start with a copy of a
machine file that matches closely with your target machine. Custom
machine files should not be pushed to the main repository.

Ubuntu 20.04 LTS
~~~~~~~~~~~~~~~~

The following procedure has been tested for an Ubuntu 20.04 LTS system:

.. code:: bash

   # install dependencies
   # openmpi is installed implicitly by the hdf5 install
   sudo apt-get update
   install cmake build-essentials libhdf5-openmpi-dev

   # make a bin directory
   mkdir bin
   cd bin
   # configure and build
   cmake ..
   cmake -j --build .
   # run unit and regression tests
   ctest -LE performance
   # run performance tests
   ctest -L performance

OLCF Summit (Power9+Volta)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Last verified 01 Feb 2021.

Common environment
^^^^^^^^^^^^^^^^^^

Load recommended modules:

.. code:: bash

   # setup environment
   $ module restore system
   $ module load cuda gcc cmake python hdf5

   # on 01 Aug 2021 that results the following version
   $ module list

   Currently Loaded Modules:
     1) hsi/5.0.2.p5    4) darshan-runtime/3.1.7   7) gcc/6.4.0                     10) spectrum-mpi/10.3.1.2-20200121
     2) xalt/1.2.1      5) DefApps                 8) cmake/3.18.2                  11) hdf5/1.10.4
     3) lsf-tools/2.0   6) cuda/10.1.243           9) python/3.6.6-anaconda3-5.3.0

Load the recommended default machine configuration:

.. code:: bash

   # assuming PARTHENON_ROOT has been set to the Parthenon folder as mentioned above
   $ export MACHINE_CFG=${PARTHENON_ROOT}/cmake/machinecfg/Summit.cmake

Build code
^^^^^^^^^^

Cuda with MPI
^^^^^^^^^^^^^

.. code:: bash

   # configure and build. Make sure to build in an directory on the GPFS filesystem if you want to run the regression tests because the home directory is not writeable from the compute nodes (which will result in the regression tests failing)
   $ mkdir build-cuda-mpi && cd build-cuda-mpi
   $ cmake ${PARTHENON_ROOT}
   $ make -j 8

   # !!!! The following commands are exepected to be run within job (interactive or scheduled), e.g., via
   # $ bsub -W 0:30 -nnodes 1 -P YOURPROJECTID -Is /bin/bash
   # and make sure to also load the module above, i.e.,
   # $ module load cuda gcc cmake/3.18.2 python hdf5

   # run all MPI regression tests (execute from within the build folder)
   $ ctest -L regression -LE mpi-no

   # Manually run a simulation (here using 1 node with 6 GPUs and 1 MPI processes per GPU for a total of 6 processes (ranks)).
   # Note the `-M "-gpu"` which is required to enable Cuda aware MPI.
   # Also note the `--kokkos-num-devices=6` that ensures that each process on a node uses a different GPU.
   $ jsrun -n 1 -a 6 -g 6 -c 42 -r 1 -d packed -b packed:7 --smpiargs=-gpu ./example/advection/advection-example -i ${PARTHENON_ROOT}/example/advection/parthinput.advection parthenon/time/nlim=10 parthenon/mesh/nx1=512 parthenon/mesh/nx2=512 parthenon/mesh/nx3=512 parthenon/meshblock/nx1=64 parthenon/meshblock/nx2=64 parthenon/meshblock/nx3=64 --kokkos-num-devices=6

Cuda without MPI
^^^^^^^^^^^^^^^^

.. code:: bash

   # configure and build
   $ mkdir build-cuda && cd build-cuda
   $ cmake -DMACHINE_VARIANT=cuda ${PARTHENON_ROOT}
   $ make -j8

   # Run unit tests (again assumes running within a job, e.g., via `bsub -W 1:30 -nnodes 1 -P PROJECTID -Is /bin/bash`)
   # - jsrun is required as the test would otherwise be executed on the scheduler node rather than on a compute node
   # - "off" is required as otherwise the implicit PAMI initialization would fail
   $ jsrun -n 1 -g 1 --smpiargs="off" ctest -L unit

   # run performance regression test test
   $ jsrun -n 1 -g 1 --smpiargs="off" ctest -R regression_test:advection_performance

LANL Darwin (Heterogeneous)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allocate Node
^^^^^^^^^^^^^

Darwin is a heterogeneous cluster, giving LANL developers easy access to
a wide variety of architectures. Therefore, before you do anything else,
you should allocate a node in the partition you intend to work in.
Currently any partition with either Haswell or newer x86-64 nodes
(e.g. ``general``, ``skylake-gold``, ``skylake-platinum``), or the
``power9`` partition will do.

E.g.

.. code:: bash

   $ salloc -p power9

Set-Up Environment (Optional, but Still Recommended, for Non-CUDA Builds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can import all tools you need to start building with by sourcing the
project ``.bashrc``:

.. code:: bash

   $ source /projects/parthenon-int/parthenon-project/.bashrc

This .bashrc will set the correct ``MACHINE_CFG`` file in your
environment, import an architecture-specific set of recent build tools
(currently cmake and ninja), and set Ninja as the default CMake
generator.

This step is required if you intend to build for CUDA (the default on
Power9).

Build the Code
^^^^^^^^^^^^^^

If you followed the “Set-Up Environment” section, configuration requires
0 additional arguments:

.. code:: bash

   $ cmake -S. -Bbuild

If you didn’t follow the “Set-Up Environment” section, you need to
specify the ``MACHINE_CFG`` file, as well.

.. code:: bash

   $ cmake -S. -Bbuild -DMACHINE_CFG=cmake/machinecfg/Darwin.cmake

The Darwin-specific dependencies, including compilers, system
dependencies, and python packages, are hard coded in ``Darwin.cmake``,
so you don’t need anything else in your environment.

Once you’ve configured your build directory, you can build with
``cmake --build build``.

Advanced
^^^^^^^^

LANL Employees - to understand how the project space is built out, see
https://re-git.lanl.gov/eap-oss/parthenon-project

LANL Snow (CTS-1)
~~~~~~~~~~~~~~~~~

.. _allocate-node-1:

Allocate Node
^^^^^^^^^^^^^

Snow is a LANL CTS-1 system with dual socket Broadwell Intel CPUs. You
can log in to ``sn-fey``. Nodes are allocated using SLURM.

E.g.

.. code:: bash

   $ salloc -N1

Set-Up Environment (Optional, but Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can import all tools you need to start building with by sourcing the
project ``.bashrc``:

.. code:: bash

   $ source /usr/projects/parthenon/parthenon-project/.bashrc

This .bashrc will set the correct ``MACHINE_CFG`` file in your
environment, import an architecture-specific set of recent build tools
(currently cmake and ninja), and set Ninja as the default CMake
generator.

.. _build-the-code-1:

Build the Code
^^^^^^^^^^^^^^

If you followed the “Set-Up Environment” section, configuration requires
0 additional arguments:

.. code:: bash

   $ cmake -S. -Bbuild

If you didn’t follow the “Set-Up Environment” section, you need to
specify the ``MACHINE_CFG`` file, as well.

.. code:: bash

   $ cmake -S. -Bbuild -DMACHINE_CFG=cmake/machinecfg/Snow.cmake

Parthenon is built with the Intel compilers by default on Snow. To build
with gcc, specify ``-DSNOW_COMPILER=GCC``.

The Snow-specific dependencies, including compilers, system
dependencies, and python packages, are hard coded in ``Snow.cmake``, so
you don’t need anything else in your environment.

Once you’ve configured your build directory, you can build with
``cmake --build build``.

.. _advanced-1:

Advanced
^^^^^^^^

LANL Employees - to understand how the project space is built out, see
https://re-git.lanl.gov/eap-oss/parthenon-project

LNLL RZAnsel (Homogeneous)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Last verified 04 Jan 2021.

.. _allocate-node-2:

Allocate Node
^^^^^^^^^^^^^

`RZAnsel <https://hpc.llnl.gov/hardware/platforms/rzansel>`__ is a
homogeneous cluster consisting of 2,376 nodes with the IBM Power9
architecture with 44 nodes per core and 4 Nvidia Volta GPUs per node. To
allocate an interactive node:

E.g.

.. code:: bash

   $ lalloc 1

.. _set-up-environment-optional-but-still-recommended-for-non-cuda-builds-1:

Set-Up Environment (Optional, but Still Recommended, for Non-CUDA Builds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can import all tools you need to start building with by sourcing the
project ``.bashrc``, to be able to access /usr/gapps/parthenon_shared
you will need to be added to the parthenon group (contact @agaspar):

.. code:: bash

   $ source /usr/gapps/parthenon_shared/parthenon-project/.bashrc

This .bashrc will set the correct ``MACHINE_CFG`` file in your
environment, import an architecture-specific set of recent build tools
(currently cmake and ninja), and set Ninja as the default CMake
generator.

This step is required if you intend to build for CUDA (the default on
Power9).

.. _build-the-code-2:

Build the Code
^^^^^^^^^^^^^^

If you followed the “Set-Up Environment” section, configuration requires
0 additional arguments:

.. code:: bash

   $ cmake -S. -Bbuild

By default cmake will build parthenon with cuda and mpi support. Other
machine variants exist and can be specified by using the
``MACHINE_VARIANT`` flag. The supported machine variants include:

-  cuda-mpi
-  mpi
-  cuda

If you didn’t follow the “Set-Up Environment” section, you need to
specify the ``MACHINE_CFG`` file, as well.

.. code:: bash

   $ cmake -S. -Bbuild -DMACHINE_CFG=cmake/machinecfg/RZAnsel.cmake

The RZAnsel-specific dependencies, including compilers, system
dependencies, and python packages, are hard coded in ``RZAnsel.cmake``,
so you don’t need anything else in your environment.

Once you’ve configured your build directory, you can build with
``cmake --build build``.

.. _advanced-2:

Advanced
^^^^^^^^

LANL Employees - to understand how the project space is built out, see
https://xcp-gitlab.lanl.gov/eap-oss/parthenon-project

LLNL RZAnsel (Power9+Volta)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Last verified 02 Sept 2020.

.. _common-environment-1:

Common environment
^^^^^^^^^^^^^^^^^^

.. code:: bash

   # setup environment
   $ module restore system
   $ module load cuda gcc/7.3.1

   # on 02 Sept 2020 that results the following version
   $ module list

   Currently Loaded Modules:
     1) StdEnv (S)   2) cuda/10.1.243   3) gcc/7.3.1   4) spectrum-mpi/rolling-release

     Where:
      S:  Module is Sticky, requires --force to unload or purge

.. _cuda-with-mpi-1:

Cuda with MPI
^^^^^^^^^^^^^

.. code:: bash

   # configure and build. Make sure to build in an directory on the GPFS filesystem if you want to run the regression tests because the home directory is not writeable from the compute nodes (which will result in the regression tests failing)
   $ mkdir build-cuda-mpi && cd build-cuda-mpi
   # note that we do not specify the mpicxx wrapper in the following as cmake automatically extracts the required include and linker options
   $ cmake -DPARTHENON_DISABLE_HDF5=On -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_POWER9=True -DKokkos_ENABLE_CUDA=True -DKokkos_ARCH_VOLTA70=True -DCMAKE_CXX_COMPILER=${PWD}/../external/Kokkos/bin/nvcc_wrapper ..
   $ make -j

   # The following commands are exepected to be run within job (interactive or scheduled)

   # Make sure that GPUs are assigned round robin to MPI processes
   $ export KOKKOS_NUM_DEVICES=4

   # run all MPI regression tests
   $ ctest -L regression -LE mpi-no

   # manually run a simulation (here using 1 node with 4 GPUs and 1 MPI processes per GPU and a total of 2 processes (ranks))
   # note the `-M "-gpu"` which is required to enable Cuda aware MPI
   # also note the `--kokkos-num-devices=1` that ensures that each process on a node uses a different GPU
   $ jsrun -p 2 -g 1 -c 20 -M "-gpu" ./example/advection/advection-example -i ../example/advection/parthinput.advection parthenon/time/nlim=10 parthenon/mesh/nx1=128 parthenon/mesh/nx2=64 parthenon/mesh/nx3=64 parthenon/meshblock/nx1=32 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=32 --kokkos-num-devices=1 | tee 2.out

.. _cuda-without-mpi-1:

Cuda without MPI
^^^^^^^^^^^^^^^^

.. code:: bash

   # configure and build
   $ mkdir build-cuda && cd build-cuda
   $ cmake -DCMAKE_BUILD_TYPE=Release -DMACHINE_CFG=${PARTHENON_ROOT}/cmake/machinecfg/Summit.cmake -DMACHINE_VARIANT=cuda -DPARTHENON_DISABLE_MPI=On ${PARTHENON_ROOT}
   $ make -j10

   # run unit tests (assumes running within a job, e.g., via `bsub -W 1:30 -nnodes 1 -P PROJECTID -Is /bin/bash`)
   # - jsrun is required as the test would otherwise be executed on the scheduler node rather than on a compute node
   # - "off" is required as otherwise the implicit PAMI initialization would fail
   $ jsrun -n 1 -g 1 --smpiargs="off" ctest -L unit

   # run convergence test
   $ jsrun -n 1 -g 1 --smpiargs="off" ctest -R regression_test:advection_performance
