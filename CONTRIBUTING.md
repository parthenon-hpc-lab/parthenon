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


### Adding Regression Tests

Parthenon uses ctest to control all testing from cmake. A python script is also available 
for help analysing the results of a given test. As such adding regression tests to 
cmake consists of two parts. 

#### Creating a regression test

Each regression test should have its own folder in /tst/regression/test_suites. Lets assume
we want to add a test which we will call foo_test. We will begin by adding a folder named
foo_test to /tst/regression/test_suites. Within that folder there must exist at least two files.

1. test_suites/foo_test/\__init__.py
2. test_suites/foo_test/foo_test.py

The \__init__.py file is left empty, it is used to notify python that it is allowed to
import the contents from any python file present in the foo_test folder. 

The second file foo_test.py **must** have the same name as the folder it is located in. The
foo_test.py folder **must** contain an Analyze function. The analyze function is responsible for
checking the output of the driver (foo_driver) and ensuring it has passed. It is called once the 
foo_driver has been executed. 

All regression tests are run using the `run_test.py` script located in /test/regression. The reason
for the python script is to ensure that after the test has been run it meets the appropriate 
test criteria. 

The `run_test.py` **requires** three different files be specified:

1. the driver - the binary that is to be executed for the test
2. the input deck - parthenon input file with run parameters
3. the test folder - the folder in parthenon/tst/regression/test_suites containing test files 

```
./run_test.py --test_dir test_suites/foo_test --driver location_of_foo_driver  --driver_input location_of_foo_input_deck
```

Assuming parthenon was compiled in parthenon/build one could run the calculate_pi example with the python script `run_test.py` from /tst/regression using

```
./run_test.py --test_dir test_suites/calculate_pi --driver ../../build/example/calculate_pi/parthenon-example --driver_input test_suites/calculate_pi/parthinput.regression
```

More options can be passed to `run_test.py` but those three **must** be specified. For options
that are supported you can call `run_test.py -h`

Note that running more than a single test with `run_test.py` is intentinally not supported, Running
groups of tests will be handled by ctest. 

#### Integrating the regression test with CMake

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

#### Running ctest 

If the above steps have been executed, the tests will automatically be built anytime the code is
compiled. The tests can be run by calling ctest directly from the build directory. Individual tests 
or groups of tests can be run by using ctest regular expression matching, or making use of labels. 

All regression tests have the following name format:

regression_test:foo_test

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



