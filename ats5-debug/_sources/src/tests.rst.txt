.. _tests:

How to add tests to Parthenon
==============================

Unit Tests
-----------

Unit tests are straightforward to implement. Open the ``tst/unit``
directory to see the current test suites. You may either add a new
file to this directory (and the associated ``CMakeLists.txt`` file, or
extend an existing file.

Parthenon uses the `Catch2`_ unit test framework. Tests are typically
written in the following format:

.. code:: cpp

   TEST_CASE("Name", "[category1][category2]") {
     GIVEN("Set up code") {
       // some code
       WHEN("Trigger") {
         THEN("Condition") {
           REQUIRE(some_bool_expression);
         }
       }
     }
   }

See the Catch documentation for more details.

.. _Catch2: https://github.com/catchorg/Catch2/tree/v2.x

Regression Tests
-----------------

The regression test infrastructure is more complicated, and our
regression test infrastructure is built on a mix of Python and
CMake. Each test is defined by a *test suite*. You can find the test
suites in the ``tst/regression/test_suites`` directory. Each test
suite is a Python module that defines a ``TestCase`` class, which
inherits from the abstract base class provided by the
``utils.test_case`` module included in the test suite. A ``TestCase``
class must implement the following methods:

* ``Prepare(self, parameters, step)`` is the python code which sets up
  a simulation run. It modifies an included input deck for a given
  test, based on the test design. The ``parameters`` input contains a
  list of command line arguments that should modify the parthenon
  run. These are passed in to the test infrastructure via CMake
  (described below). The ``step`` argument is an integer. It is used
  for regression tests that require multiple simulation runs, such as
  a convergence test.

* ``Analyze(self, parameters)`` is the post-processing step that
  checks whether or not the test passed. Some tests compare to gold
  files (described further below) and some simply compare to a known
  solution.

A test suite needs to have not only the python file containing the
``TestCase`` class, but an empty ``__init__.py`` file to match the
Python module API.

After adding a module, you must also modify the file
``tst/regression/CMakeLists.txt``. In particular for a new regression
test, you must add a set of arguments like these:

::

   list(APPEND TEST_DIRS name_of_folder )
   list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
   list(APPEND TEST_ARGS args_to_pass_to_run_test.py )
   list(APPEND EXTRA_TEST_LABELS "my-awesome-test")

The first argument specifies the name of the folder containing the new
python module for the new test. The second argument specifies number
of MPI ranks if the test should be run with MPI (specify 1 if
not). The third argument specifies arguments to pass to your test
suite, for example

::

   "--driver ${PROJECT_BINARY_DIR}/example/advection/advection-example --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/advection_performance/parthinput.advection_performance --num_steps 4"

which would specify which application to run for the test, as well as the input deck and the number of steps.

The final argument specifies labels attached to the test for use with
CTest.

Gold Files
-----------

Many tests use so-called *gold files*, files are files containing
known results to compare against. Parthenon bundles its gold files as
part of releases. These files are automatically downloaded and are
located in ``tst/regression/gold_standard``. To add a new gold file
(or update an old one), place it in this directory.

To make the new (or updated) test official, you must add it to the
official test suite. First update
``tst/regression/gold_standard/README.rst`` and add a new version of
the test suite, corresponding to the commit where you added the
relevant test code and an explanation for why the gold files needed to
change. Then run the script ```make_tarball.sh`` as

::

   bash make_tarball.sh NEW_VERSION

where ``NEW_VERSION`` is the new version of the gold files (not
necessarily tied to the version of the code release). You can then ask
a maintainer to create a new goldfile release and attach the resultant
tarball to the release.

As a sanity check, Parthenon checks against the ``sha512`` hash of the
tarball. The make tarball script will output the hash. The new version
and new hash must be set as the default values of the
``REGRESSION_GOLD_STANDARD_VER`` and ``REGRESSION_GOLD_STANDARD_HASH``
in the top level ``CMakeLists.txt`` file.
