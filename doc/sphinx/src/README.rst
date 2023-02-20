Parthenon documentation
=======================

This file provides an overview of different (not necessarily all)
features in Parthenon and how to use them.

Building parthenon
------------------

See the :ref:`build doc <building.md>` for details on building Parthenon
for specific systems.

Description of examples
-----------------------

-  `Calculate π <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/calculate_pi>`_
-  `Average face-centered variables to cell
   centers <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/face_fields>`_
-  `Stochastic subgrid <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/stochastic_subgrid>`_ Performs a
   random amount of work per cell (drawn from a power law distribution)

Short feature descriptions
--------------------------

Automated tests
~~~~~~~~~~~~~~~

Regression and convergence tests that cover the majority of features are
based on the `Advection example <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/advection-example>`__ and
defined in the
`advection-convergence <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/tst/regression/test_suites/advection_convergence>`__
and `output_hdf5 <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/tst/regression/test_suites/output_hdf5>`__ test
suites.

The tests currently cover

- advection of wave in x, y, and z direction
  as well oblique to the *static* grid for different resolutions to
  demonstrate first order convergence (see
  ``tst/regression/outputs/advection_convergence/advection-errors.png``
  file in the build directory after running the test)
- Advection of a smoothed sphere at an angle on a *static* grid, on a *static* grid a
  twice the resolution, and with *AMR* covering the sphere at the
  effective higher resolution
- Advection of a sharp sphere at an angle
  with *AMR* writing hdf5 output and comparing against a gold standard output.

To execute the tests run, e.g.,

.. code:: bash

   # from within the build directory (add -V fore more detailed output)
   ctest -R regression

The gold standard files (reference solutions) used in the regression
tests should automatically be downloaded during the configure phase.
Alternatively, you can download them as a release from
`GitHub <https://github.com/parthenon-hpc-lab/parthenon/releases/>`__ and extract the
contents of the archive to
``PARTHENON_ROOT/tst/regression/gold_standard`` directory. Make sure to
get the correct version matching your source (stored in the
``REGRESSION_GOLD_STANDARD_VER`` CMake variable). Note: If you results
are (slightly) different, that may stem from using different
compiler/optimization options.

In case you adds new tests that require reference data just put all file
in the ``PARTHENON_ROOT/tst/regression/gold_standard`` directory and
either

- increase the version integer by one (both in the
  ``PARTHENON_ROOT/tst/regression/gold_standard/current_version`` file and
  in the ``PARTHENON_ROOT/CMakeLists.txt`` file), or
- configure with
  ``REGRESSION_GOLD_STANDARD_SYNC=OFF``. The former is considered the
  safer option as it prevents accidental overwriting of those files during
  configure (as ``REGRESSION_GOLD_STANDARD_SYNC`` is ``ON`` by default).
  In the pull request of the suggested changes we will then update the
  official gold standard release file and appropriate hash prior to
  merging.

Usage in downstream codes
^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest/recommended way to reuse the ``Parthenon`` regression test
infrastructure in downstream codes is to adapt a similar directory
structure. The following steps have been tested to work when
``Parthenon`` is built from source in the downstream project
(specifically, the ``Parthenon`` source is expected to be located in the
``external/parthenon`` folder in the project’s root directory) :

1. Add the following to the downstream root ``CMakeLists.txt`` after
   ``Parthenon`` has been included:

::

   include(CTest)
   add_subdirectory(tst/regression)

.. note::
   If the ``Parthenon`` regression tests should also be integrated
   in the downstream testing, the binary output directory should only be
   changed (e.g., via the ``CMAKE_RUNTIME_OUTPUT_DIRECTORY`` variable)
   *after* ``Parthenon`` has been included. Otherwise the paths to the
   ``Parthenon`` regression test binaries will be wrong.

2. Create the following directories and files in the project folder (for an example
   ``my_first_test`` test):

::

   tst/
       regression/
           CMakeLists.txt
           test_suites/
               __init__.py  # <-- empty file
               my_first_test/
                   __init.py__  # <-- empty file
                   my_first_test.py

3. Contents of ``tst/regression/CMakeLists.txt``

::


   # import Parthenon setup_test_serial and setup_test_parallel
   include(${PROJECT_SOURCE_DIR}/external/parthenon/cmake/TestSetup.cmake)

   setup_test_serial("my_first_test" "--driver /path/to/downstream_binary \
     --driver_input ${PROJECT_SOURCE_DIR}/inputs/test_input_file.in --num_steps 3" "my_custom_test_label")

The same options for ``setup_test_serial`` and ``setup_test_parallel``
as described in Parthenon `here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/tst/regression/CMakeLists.txt>`__
and `here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/cmake/TestSetup.cmake>`__ apply.

4. ``my_first_test.py`` contains the same logic as any other test in Parthenon, see a
   `simple <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/tst/egression/test_suites/advection_outflow/advection_outflow.py>`__
   or more
   `complicated <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/tst/egression/test_suites/advection_outflow/advection_outflow.py>`__
   example.

5. Now ``my_first_test`` should be automatically executed when
   running ``ctest`` from the build directory.

ParthenonManager
~~~~~~~~~~~~~~~~

This class provides a streamlined capability to write new applications
by providing a simple interface to initialize and finalize a simulation.
It’s usage is straightforward and demonstrated in the π
`example <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/calculate_pi/calculate_pi.cpp>`__.

Initialization is mandatory and takes care of (including sanity checks)

1. initializing MPI (if enabled)
2. initializing Kokkos (including device setup)
3. parsing command line arguments and parameter input file
4. ``ProcessPackages`` Constructs and returns a ``Packages_t`` object
   that contains a listing of all the variables and their metadata
   associated with each package.

Application can chose between a single and double stage initialization:

- Single stage: ``ParthenonInit(int argc, char *argv[])`` includes steps 1-5 above.
- Double stage: ``ParthenonInitEnv(int argc, char *argv[])``
  includes steps 1-3 and ``ParthenonInitPackagesAndMesh()`` includes steps
  4 and 5. This double stage setup allows, for example, to control the
  package's behavior at runtime by setting the problem generator based on
  a variable in the input file.

User-specified internal functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During a simulation, Parthenon calls a number of default internal
functions whose behavior can be redefined by an application. Currently,
these functions are, by class:

Mesh
^^^^

-  ``InitUserMeshData``
-  ``PreStepUserWorkInLoop``
-  ``PostStepUserWorkInLoop``
-  ``UserWorkAfterLoop``

MeshBlock
^^^^^^^^^

-  ``InitApplicationMeshBlockData``
-  ``InitMeshBlockUserData``
-  ``ProblemGenerator``
-  ``UserWorkBeforeOutput``

To redefine these functions, the user sets the respective function
pointers in the ApplicationInput member app_input of the
ParthenonManager class prior to calling ``ParthenonInit``. This is
demonstrated in the ``main()`` functions in the examples.

Note that the ``ProblemGenerator``\ s of ``Mesh`` and ``MeshBlock`` are
mutually exclusive. Moreover, the ``Mesh`` one requires
``parthenon/mesh/pack_size=-1`` during initialization, i.e., all blocks
on a rank need to be in a single pack. This allows to use MPI reductions
inside the function, for example, to globally normalize quantities. The
``parthenon/mesh/pack_size=-1`` exists only during problem
inititalization, i.e., simulations can be restarted with an arbitrary
``pack_size``. For an example of the ``Mesh`` version, see the `Poisson
example <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/example/poisson/parthenon_app_inputs.cpp>`__.

Error checking
~~~~~~~~~~~~~~

Macros for causing execution to throw an exception are provided
`here <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/src/utils/error_checking.hpp>`__

- ``PARTHENON_REQUIRE(condition, message)`` exits if the condition does not evaluate to true.
- ``PARTHENON_REQUIRE_THROWS(condition, message)`` throws a ``std::runtime_error`` exception if the condition does not evaluate to true.
- ``PARTHENON_FAIL(message)`` always exits.
- ``PARTHENON_THROW(message)`` throws a runtime error.
- ``PARTHENON_DEBUG_REQUIRE(condition, message)`` exits if the condition does not evaluate to true when in debug mode.
- ``PARTHENON_DEBUG_REQUIRE_THROWS(condition, message)`` throws if the condition does not evaluate to true when in debug mode.
- ``PARTHENON_DEBUG_FAIL(message)`` always exits when in debug mode.
- ``PARTHENON_DEBUG_THROW(message)`` throws a runtime error when in debug mode.

All macros print the message, and filename and line number where the
macro is called. PARTHENON_REQUIRE also prints the condition. The macros
take a ``std::string``, a ``std::stringstream``, or a C-style string. As
a rule of thumb:

- Use the exception throwing versions in non-GPU,
  non-performance critical code.
- On GPUs and in performance-critical
  sections, use the non-throwing versions and give them C-style strings.

Developer guide
~~~~~~~~~~~~~~~

Please see the :ref:`full development guide <development.md>` on how to use
Kokkos-based performance portable abstractions available within
Parthenon and how to write performance portable code.

State Management
~~~~~~~~~~~~~~~~

:ref:`Full Documentation <interface/state.md>`

Parthenon provides a convenient means of managing simulation data.
Variables can be registered with Parthenon to have the framework
automatically manage the field, including updating ghost cells,
prolongation, restriction, and I/O.

Application Drivers
~~~~~~~~~~~~~~~~~~~

A description of the Parthenon-provided classes that facilitate
developing the high-level functionality of an application (e..g. time
stepping) can be found :ref:`here <driver.md>`.

Adaptive Mesh Refinement
~~~~~~~~~~~~~~~~~~~~~~~~

A description of how to enable and extend the AMR capabilities of
Parthenon is provided :ref:`here <amr.md>`.

Tasks
~~~~~

The tasking capabilities in Parthenon are documented
:ref:`here <tasks.md>`.

Outputs
~~~~~~~

Check :ref:`here <outputs.md>` for a description of how to get data out of
Parthenon and how to visualize it.

MeshBlockDatas and MeshBlockData Iterators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`here <interface/containers.md>` for a description of containers,
container iterators, and variable packs.

Index Shape and Index Range
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A description of mesh indexing classes :ref:`here <mesh/domain.md>`.

Coordinates
~~~~~~~~~~~

Accessing coordinate information on each block is described
:ref:`here <coordinates.md>`. Currently only uniform Cartesian coordinates
are supported in Parthenon but uniform Spherical and cylindrical
coordinates specified at compile time are forthcoming.

Input file parameter
~~~~~~~~~~~~~~~~~~~~

An overview of input file parameters :ref:`here <inputs.md>`

Note that all parameters can be overridden (or new parameters added)
through the command line by appending the parameters to the launch
command. For example, the ``refine_tol`` parameter in the
``<parthenon/refinement0>`` block in the input file can be changed by
appending ``parthenon/refinement0/refine_tol=my_new_value`` to the
launch command (e.g.,
``srun ./myapp -i my_input.file parthenon/refinement0/refine_tol=my_new_value``).
This similarly applies to simulations that are restarted.

Global reductions
~~~~~~~~~~~~~~~~~

Global reductions are a common need for downstream applications and can
be accomplished within Parthenon’s task-based execution as described
:ref:`here <reductions.md>`.

Solvers
~~~~~~~

Solvers are still a work in progress in Parthenon, but some basic
building blocks are described :ref:`here <solvers.md>`.
