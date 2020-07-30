# Parthenon Development Guide

The following sections cover the current guidelines that all developers are expected
to try to follow.
As with all development these guidelines are not set in stone but subject to discussion
and revision any time as necessary.

## General workflow

### Planning/requesting/announcing changes

If you discover a bug, would like to see a new feature, would like to implement
a new feature, or have any other item/idea that you would like to share
and/or discuss open an issue.

This helps us to keep track of who is working on what and prevent duplicated efforts
and/or allows to pool resources early on.

Use GitHub labels as appropriate and feel free to directly add/tag people to the issue.

### Contributing code

In order to keep the main repository in order, everyone is encouraged to create feature
branches starting with their username, followed by a "/", and ending with a brief
description, e.g., "username/add_feature_xyz".
Working on branches in private forks is also fine but not recommended (as the automated
testing infrastructure will then first work upon opening a pull request).

Once all changes are implemented or feedback by other developers is required/helpful
open a pull request again the master branch of the main repository.

In that pull request refer to the issue that you have addressed.

If your branch is not ready for a final merge (e.g., in order to discuss the 
implementation), mark it as "work in progress" by prepending "WIP:" to the subject.

### Merging code

In order for code to be merged into master it must

- obey the style guide (test with CPPLINT)
- pass the linting test (test with CPPLINT)
- pass the formatting test (see "Formatting Code" below)
- pass the existing test suite
- have at least one approval by one member of each physics/application code
- include tests that cover the new feature (if applicable)
- include documentation in the `doc/` folder (feature or developer; if applicable)

The reviewers are expected to look out for the items above before approving a merge
request.

#### Linting Code
cpplint will automatically run as part of the cmake build on individual source
files when building the default target. You can run the lint explicitly by
building the `lint` target.
```
cmake --build . --target lint
```

If you do not want to the code to be linted as part of the default target, you
can disable that behavior with the `PARTHENON_LINT_DEFAULT` cmake option.
```
cmake -DPARTHENON_LINT_DEFAULT=OFF .
```

#### Formatting Code
We use clang-format to automatically format the code. If you have clang-format installed
locally, you can always execute `make format` or `cmake --build . --target format` from
your build directory to automatically format the code.

If you don't have clang-format installed locally, our "Hermes" automation can always
format the code for you. Just create the following comment in your PR, and Hermes will
format the code and automatically commit it:
```
@par-hermes format
```

After Hermes formats your code, remember to run `git pull` to update your local tracking
branch.

**WARNING:** Due to a limitation in GitHub Actions, the "Check Formatting" CI will not
run, which will block merging. If you don't plan on making any further commits, you or a
reviewer can run ["./scripts/retrigger-ci.sh"](scripts/retrigger-ci.sh) with your branch
checked out to re-run the CI.

If you'd like Hermes to amend the formatting to the latest commit, you can use the
`--amend` option. WARNING: The usual caveats with changing history apply. See:
https://mirrors.edge.kernel.org/pub/software/scm/git/docs/user-manual.html#problems-With-rewriting-history
```
@par-hermes format --amend
```

You will remain the author of the commit, but Hermes will amend the commit for you with
the proper formatting.
Remember - this will change the branch history from your local commit, so you'll need to
run something equivalent to
`git fetch origin && git reset --hard origin/$(git branch --show-current)` to update your
local tracking branch.

## Test suite

### Continuous testing/integration environment

Commits pushed to any branch of this repository is automatically tested by
two CI pipelines.

The first pipeline focuses on correctness targeting code style, formatting, as well
as unit and regression tests.
It is executed through a repository [mirror](https://gitlab.com/pgrete/parthenon) on GitLab
on a machine with an Intel Xeon E5540 (Broadwell) processor and Nvidia GeForce GTX 1060 (Pascal) GPU.
The Dockerfile for the CI runner can be found [here](scripts/docker/Dockerfile.nvcc) and the
pipeline is configured through [.gitlab-ci.yml](.gitlab-ci.yml).
The current tests span MPI and non-MPI configurations on CPUs (using GCC) and GPUs (using Cuda/nvcc).

The second pipeline focuses on performance regression.
It is executed through a (different) repository [mirror](https://gitlab.com/theias/hpc/jmstone/athena-parthenon/parthenon-ci-mirror)
using runners provided by the IAS.
The runners have Intel Xeon Gold 6148 (Skylake) processors and Nvidia V100 (Volta) GPUs.
Both the environment and the pipeline are configoures through [.gitlab-ci-ias.yml](.gitlab-ci-ias.yml).
The current tests span uniform grids on GPUs (using Cuda/nvcc).
Note, in order to integrate this kind of performance regression test with CMake
follow the instructions [below](#integrating-the-regression-test-with-cmake) *and* add the
`perf-reg` label to the test (see bottom of the regression
[CMakeLists.txt](tst/regression/CMakeLists.txt)).

### Adding Tests

Five categories of tests have been identified in parthenon, and they are
located in their respective folders in the `tst` folder. 

1. Unit testing
2. Integration testing
3. Performance testing
4. Regression testing
5. Style

Parthenon uses ctest to manage the different tests. Cmake labels are attached
to each test to provide control over which group of tests should be executed.
Any test added within the `tst/unit`, `tst/integration`, `tst/performance` or
`tst/regression` test folders will automatically be associated with the
appropriate label.
 
When possible a test should be designed to fall into only one of these
categories. For instance a unit test should not also have a performance
component. If such a case occurs, it is better to split the test into two
separate tests, one that tests the performance and one that tests the
correctness.

#### Adding Regression Tests

A python script is available for help analysing the results of each regression
test. As such adding regression tests to cmake consists of two parts. 

##### Creating a regression test

Each regression test should have its own folder in /tst/regression/test\_suites. Lets assume
we want to add a test which we will call foo\_test. We will begin by adding a folder named
foo\_test to /tst/regression/test\_suites. Within that folder there must exist at least two files.

1. test\_suites/foo\_test/\_\_init\_\_.py
2. test\_suites/foo\_test/foo\_test.py

The \_\_init\_\_.py file is left empty, it is used to notify python that it is allowed to
import the contents from any python file present in the foo\_test folder. 

The second file foo\_test.py **must** have the same name as the folder it is located in. The
foo\_test.py folder **must** contain a class called TestCase which inherets from TestCaseAbs. The
TestCase class **must** contain an Analyze and Prepare method. The prepare method can be used to 
execute tasks before the driver is called such a file prepartion or overwriting arguments that
are passed in through the input deck file. The analyze method is responsible for
checking the output of the driver (foo\_driver) and ensuring it has passed. It is called once the 
foo\_driver has been executed. 

Here is a base template for foo\_test.py:

```
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True

class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self,parameters):
        return parameters

    def Analyse(self,parameters):
        analyze_status = True
        return analyze_status

```

All regression tests are run using the `run_test.py` script located in /test/regression. The reason
for the python script is to ensure that after the test has been run it meets the appropriate 
test criteria. 

The `run_test.py` **requires** three different files be specified:

1. the driver - the binary that is to be executed for the test
2. the input deck - parthenon input file with run parameters
3. the test folder - the folder in parthenon/tst/regression/test\_suites containing test files 

```
./run_test.py --test_dir test_suites/foo_test --driver location_of_foo_driver  --driver_input location_of_foo_input_deck
```

Assuming parthenon was compiled in parthenon/build one could run the calculate\_pi example with the python script `run_test.py` from /tst/regression using

```
./run_test.py --test_dir test_suites/calculate_pi --driver ../../build/example/calculate_pi/parthenon-example --driver_input test_suites/calculate_pi/parthinput.regression
```

More options can be passed to `run_test.py` but those three **must** be specified. For options
that are supported you can call `run_test.py -h`

Note that running more than a single test with `run_test.py` is intentinally not supported, Running
groups of tests will be handled by ctest. 

##### Integrating the regression test with CMake

At this point CMake and ctest know nothing about your dandy new regression test. To integrate the
test with cmake it needs to be added to the CMakeLists.txt file in parthenon/tst/regression. 
Essentially, the command for running `run_test.py` must be called from within cmake. 

```
list(APPEND TEST_DIRS foo_test)
list(APPEND TEST_ARGS "--driver location_of_foo_driver --driver_input Location_of_foo_input_deck")
```

For the calculate pi example shown above this consists of adding parameters to two cmake lists

```
list(APPEND TEST_DIRS calculate_pi)
list(APPEND TEST_ARGS "--driver ${CMAKE_BINARY_DIR}/example/calculate_pi/parthenon-example --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/calculate_pi/parthinput.regression")
```

NOTE: By default all regression tests added to these lists will be run in serial and in parallel
with mpi. The number of mpi processors used is by default set to 4. This default can be adjusted
by changing the cmake variable NUM\_MPI\_PROC\_TESTING. The number of OpenMP threads is by default set
to 1 but can be adjusted in the driver input file deck. 

##### Running ctest 

If the above steps have been executed, the tests will automatically be built anytime the code is
compiled. The tests can be run by calling ctest directly from the build directory. Individual tests 
or groups of tests can be run by using ctest regular expression matching, or making use of labels. 

All regression tests have the following name format:

regression\_test:foo\_test

So for the pi example we will see the following output:

```
ctest -R regression

    Start 17: regression_test:calculate_pi
1/1 Test #17: regression_test:calculate_pi .....   Passed    1.44 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
regression    =   1.44 sec*proc (1 test)

Total Test time (real) =   1.47 sec
```

#### Including Tests in Code Coverage Report

Tests are not automatically included in the coverage report. To include a unit or integration test in the coverage
report the tag [coverage] should be added to the catch2 `TEST_CASE` macro. E.g.

```
TEST_CASE("test description", "[unit][coverage]"){
...
}
```

To add a regression test to the coverage report an attribute of the parameters argument can be 
altered. E.g.

```
class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self,parameters):
        parameters.coverage_status = "only-coverage"
        return parameters
```

The coverage status argument can take 3 different settings:

1. **only-regression** - will not run if with coverage, is the default
2. **both** - will run with or without coverage
3. **only-coverage** - will only run if coverage is specified

##### Creating and Uploading Coverage Report

To create a coverage report cmake should be run with both the CMAKE_BUILD_TYPE
flag set to debug, and the CODE_COVERAGE flag set to On. 

```
  mkdir build
  cd build
  cmake -DCODE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug ../
  make coverage
  make coverage-upload
``` 

Fine grained control of where the coverage reports are placed can be specified
with COVERAGE_PATH, COVERAGE_NAME, which represent the path to the coverage
reports and the directory where they will be placed. The default location is in
a folder named coverage in the build directory. 


