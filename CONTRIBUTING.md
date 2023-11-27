# Parthenon Development Guide

The following sections cover the current guidelines that all developers are expected
to try to follow.
As with all development these guidelines are not set in stone but subject to discussion
and revision any time as necessary.

1. [General Workflow](#general-workflow)
    - [Planning/Requesting/Announcing Changes](#planningrequestingannouncing-changes)
    - [Sumary of Branching Model and Versioning](#summary-of-branching-model-and-versioning)
    - [Contributing Code](#contributing-code)
    - [Merging Code](#merging-code)
        * [Merging Code from a Fork](#merging-code-from-a-fork)
    - [Linting Code](#linting-code)
    - [Formatting Code](#formatting-code)
2. [Test Suite](#test-suite)
    - [Continuous Testing/Integration Environment](#continuous-testingintegration-environment)
    - [Adding Tests](#adding-tests)
        * [Adding Regression Tests](#adding-regression-tests)
            - [Creating a Regression Test](#creating-a-regression-test)
            - [Integrating the Regression Test with CMake](#integrating-the-regression-test-with-cmake)
            - [Running ctest](#running-ctest)
        * [Adding Performance Regression Tests](#adding-performance-regression-tests)
        * [Including Tests in Code Coverage Report](#including-tests-in-code-coverage-report)
        * [Creating and Uploading Coverage Report](#creating-and-uploading-coverage-report)

## General Workflow

### Planning/Requesting/Announcing Changes

If you discover a bug, would like to see a new feature, would like to implement
a new feature, or have any other item/idea that you would like to share
and/or discuss open an issue.

This helps us to keep track of who is working on what and prevent duplicated efforts
and/or allows to pool resources early on.

Use GitHub labels as appropriate and feel free to directly add/tag people to the issue.

### Summary of Branching Model and Versioning

Only a single main branch called `develop` exists and all PRs should be merged
into that branch.
Individual versions/releases are tracked by tags.

We aim at creating a new release every 6 months.
The decision on creating a new release is made during the bi-weekly calls.
Following steps need to be done for a new release:

- Create a new tag for that version using a modified [calender versioning](https://calver.org/) scheme.
Releases will be tagged `vYY.0M` i.e., short years and zero-padded months.
- Update the version in the main `CMakeLists.txt`.
- Update the `CHANGELOG.md` (i.e., add a release header and create new empty
categories for the "Current develop" section.
- Sent a mail to the mailing list announcing the new release.

### Contributing Code

In order to keep the main repository in order, everyone is encouraged to create feature
branches starting with their username, followed by a "/", and ending with a brief
description, e.g., "username/add\_feature\_xyz".
Working on branches in private forks is also fine but not recommended (as the automated
testing infrastructure will then first work upon opening a pull request).

Once all changes are implemented or feedback by other developers is required/helpful
open a pull request again the `develop` branch of the main repository.

In that pull request refer to the issue that you have addressed.

If your branch is not ready for a final merge (e.g., in order to discuss the
implementation), mark it as "work in progress" by prepending "WIP:" to the subject.

### Merging Code

In order for code to be merged into `develop` it must

- obey the style guide (test with CPPLINT)
- pass the linting test (test with CPPLINT)
- pass the formatting test (see "Formatting Code" below)
- pass the existing test suite
- have at least one approval by one member of each physics/application code
- include tests that cover the new feature (if applicable)
- include documentation in the `doc/` folder (feature or developer; if applicable)
- include a brief summary in `CHANGELOG.md`

The reviewers are expected to look out for the items above before approving a merge
request.

#### Merging Code from a Fork

PRs can opened as usual from forks.
Unfortunately, the CI will not automatically trigger for forks. This is for security
reasons. As a workaround, in order to trigger the CI, a local branch will need to be created
on Parthenon first. The forked code can then be merged into the local branch on
Parthenon. At this point when a new merge request is opened from the local branch
to the develop branch it will trigger the CI.
Someone of the Parthenon core team will take care of the work around once a PR from a fork.
No extra work is required from the contributor.

The workaround workflow for the Parthenon core developer may look like
(from a local Parthenon repository pulling in changes from a `feature-A` branch in a fork):

$ git remote add external-A https://github.com/CONTRIBUTOR/parthenon.git
$ git fetch external-A
$ git checkout external-A/feature-A
$ git push --set-upstream origin CONTRIBUTOR/feature-A

NOTE: Any subsequent updates made to the forked branch will need to be manually pulled into the local branch.

### Linting Code
cpplint will automatically run as part of the CI. You can run the lint explicitly by
building the `lint` target.
```
cmake --build . --target lint
```

If you want the code to be linted as part of the default build target, you
can enable that behavior with the `PARTHENON_LINT_DEFAULT` cmake option.
```
cmake -DPARTHENON_LINT_DEFAULT=ON .
```

### Formatting Code
We use clang-format to automatically format the C++ code. If you have clang-format installed
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

In addition to clang-format, black is used to enforce formatting on python scripts.
Running:
```
@par-hermes format
```

Will also format all the ".py" files found in the repository.

## Test Suite

### Continuous Testing/Integration Environment

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
Both the environment and the pipeline are configured through [.gitlab-ci-ias.yml](.gitlab-ci-ias.yml).
The current tests span uniform grids on GPUs (using Cuda/nvcc).
Note, in order to integrate this kind of performance regression test with CMake
follow the instructions [below](#integrating-the-regression-test-with-cmake) *and* add the
`perf-reg` label to the test (see bottom of the regression
[CMakeLists.txt](tst/regression/CMakeLists.txt)).

A third pipeline is run using LANL internal systems and is run manually when
approved, it is also scheduled to run on a daily basis on the development
branch. The internal machines use the newest IBM powerPC processors and the
NVIDIA V100 (Volta) GPUs (power9 architecture). Tests run on these systems are
primarily aimed at measuring the performance of this specific architecture.
Compilation and testing details can be found by looking in the
[.gitlab-ci-darwin.yml](.gitlab-ci-darwin.yml) file *and* the /scripts/darwin
folder. In summary, the CI is built in release mode, with OpenMP, MPI, HDF5 and
Cuda enabled. All tests are run on a single node with access to two Volta
GPUs. In addition, the regression tests are run in parallel with two mpi
processors each of which have access to their own Volta gpu. The following
tests are run with this CI: unit, regression, performance. A final note,
this CI has been chosen to also check for performance regressions. The CI
uses a GitHub application located in /scripts/python. After a successful run
of the CI a link to the performance metrics will appear as part of the parthenon
metrics status check in the pr next to the commit the metrics were recorded for.
All data from the regression tests are recorded in the parthenon wiki in a JSON file.

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

##### Creating a Regression Test

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

##### Integrating the Regression Test with CMake

At this point CMake and ctest know nothing about your dandy new regression test. To integrate the
test with cmake it needs to be added to the CMakeLists.txt file in parthenon/tst/regression.
Essentially, the command for running `run_test.py` must be called from within cmake.

```
list(APPEND TEST_DIRS foo_test)
list(APPEND TEST_PROCS "5")
list(APPEND TEST_ARGS "--driver location_of_foo_driver --driver_input Location_of_foo_input_deck")
list(APPEND EXTRA_TEST_LABELS "")
```

For the calculate pi example shown above this consists of adding parameters to four cmake lists

```
list(APPEND TEST_DIRS calculate_pi)
list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
list(APPEND TEST_ARGS "--driver ${CMAKE_BINARY_DIR}/example/calculate_pi/parthenon-example --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/calculate_pi/parthinput.regression")
list(APPEND EXTRA_TEST_LABELS "perf")
```

NOTE: The TEST\_PROCS list indicates how many processors to use when running
mpi. The cmake variable NUM\_MPI\_PROC\_TESTING can be used if you do not want
to hardcode a value, and is recommended.  By default all regression tests added
to these lists will be run in serial and in parallel with mpi. The number of
mpi processors used is by default set to 4. This default can be adjusted by
changing the cmake variable NUM\_MPI\_PROC\_TESTING. The number of OpenMP
threads is by default set to 1 but can be adjusted in the driver input file
deck. If parthenon is compiled with CUDA enabled, by default a single GPU will
be assigned to each node..

##### Integratingthe Regression Test with the Github python performance regression app

In addition to running regression tests, the Parthenon CI makes use of a Github python app to
report performance metrics on machines of interest i.e. (power9 nodes). Currently, the apps source
files are located in parthenon/scripts/python/packages/parthenon_performance_app. To add
additional tests metrics changes will need to be made to the scripts located in the folder:
parthenon/scripts/python/packages/parthenon_performance_app.  In general, the app
works by taking performance metrics that are ouput to a file when a regression test is executed.
This output is read by the app and compared with metrics that are stored in the wiki (JSON format).
The metrics are then plotted in a png file which is also uploaded to the wiki. Finally, a markdown
page is created with links to the images and is uploaded to the wiki.

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

#### Adding Performance Regression Tests

Regression tests can also be added for the purposes of testing the performance.
Adding performance tests does require understanding a bit about how the CI
infrastructure is setup. Currently, the performance tests are setup to run on
dedicated Power9 nodes that are provided by LANL, this is to ensure consistency
in the result. However, because the performance tests are running on LANL
machines it means that it is not straightforward to make the results viewable
to non LANL contributors. To make the results of the regression tests viewable
on GitHub a custom GitHub application was developed to be used with Parthenon
which we will call the Parthenon Performance Metrics App (PPMA).

The PPMA is series of python scripts located in the parthenon repository in
./scripts/python/packages/parthenon_performance_app. The PPMA has the following
responsibilities:

1. Submit a status check associated with the commit being analyzed to the
   Parthenon repository
2. Read in performance metrics from regression tests in a build directory
3. Compare the current metrics with metrics stored in the Parthenon Wiki, the
   files in the Wiki should contain performance metrics from previous runs.
4. Generate images showing the performance numbers.
5. Add images to an orphan branch in the Parthenon repository (branch name is
   figures).
6. Add new performance metrics to the appropriate performance JSON file stored
   in the Wiki. These JSON files are labeled with the following format:
   `performance_metrics_current-branch.json`, this means each branch should
   have its own JSON performance file.
7. Create or overwrite a wiki page that contains links to the performance
   figures located in the parthenon wiki repository.
8. Upload the wiki page the images and performance metrics to the Parthenon
   repository. The wiki page will be named with the following format:
   current-branch_target-branch.md
   E.g. If you were merging hotfix branch into develop the wiki page would be
   named: hotfix_develop.md, each PR should have it's own Wiki page. For performance
   tests that are not run as part of a wiki such as the develop branch is scheduled
   to run daily the current-branch and target-branch are the same: develop_develop.md.
9. Submit a status check indicating that the performance metrics are ready to
   be viewed with a link to the wiki page.

In order to add a performance test, let's call it `TEST`, the following tasks must be completed.

1. The test must be inside the folder `./tst/regression/test_suites/TEST_performance`.
   NOTE: the test folder must have `performance` in the name because
   regexp matching of the folder name is used to figure out which tests to run
   from the CI. To be explicit you can see what command is run by looking in the
   file `./scripts/darwin/build_fast.sh`, there is a line

   ```Bash
       ctest --output-on-failure -R performance -E "SpaceInstances"
   ```

2. When the test is run e.g. by calling ctest the test must output the
   performance metrics in a file called `TEST_results.txt` which will appear in the
   build directory.
3. The main PPMA file located in
   `./scripts/python/packages/parthenon_performance_app/bin/` must be edited. The
   PPMA script works by looping over the output directories of the regression
   tests, the loop exists in the main PPMA file and should have a logic block
   similar to what is shown below:

```python
for test_dir in all_dirs:
    if not isinstance(test_dir, str):
        test_dir = str(test_dir)
    if (test_dir == "advection_performance"
        or test_dir == "advection_performance_mpi"):

        figure_url, figure_file, _ = self._createFigureURLPathAndName(
                    test_dir, current_branch, target_branch
                )

        # do something...

        if create_figures:
            figure_files_to_upload.append(figure_file)
            figure_urls.append(figure_url)
```

Above you can see a branch for `advection_performance` and
`advection_performance_mpi`, because the logic is the same for both tests there is
a single if statement. You will likely need to add a separate branch for any
additional tests e.g. `dir_foo_performance`.

The `_createFigureURLPathAndName` is a convenience method for generating a figure URL
and a figure_file name. The `figure_url` will be of the form https://github.com/lanl/parthenon/blob/figures/figure_file.
Using this method is not a requirement. For instance, if you had multiple different performance figures that you
wanted to generate for a single test you could do, for example:

```python

if (test_dir == "dir_foo_performance"):
    jpeg_fig_name = "awesome.jpeg"
    figure_url1 = "https://github.com/lanl/parthenon/blob/figures/" + jpeg_fig_name

    png_fig_name = "awesome.png"
    figure_url2 = "https://github.com/lanl/parthenon/blob/figures/" + png_fig_name

    png_fig_name2 = "awesome2.png"
    figure_url3 = "https://github.com/lanl/parthenon/blob/figures/" + png_fig_name2

    # do something... need to actually generate the figures and read in performance metrics

    if create_figures:
        figure_files_to_upload.append(jpeg_fig_name)
        figure_urls.append(figure_url1)
        figure_files_to_upload.append(png_fig_name)
        figure_urls.append(figure_url2)
        figure_files_to_upload.append(png_fig_name2)
        figure_urls.append(figure_url3)
```

4. At this point we know the app knows the URLs of the figures and we know the
   figure names, however we have to actually generate the figures, this
   involves creating an analyzer, you can take a look at the AdvectionAnalyzer as
   an example
   ./scripts/python/packages/parthenon_performance_app/parthenon_performance_app/parthenon_performance_advection_analyzer.py.
   In general, the Analyzer has the following responsibilities:

        a. It must be able
           to read in results from the regression tests.

        b. It must be able to package the results in a json format.

        c. It must be able to read previous performance metrics results from the json
           files stored in the wiki. A convenience script is provided to help with this in
           the form of the parthenon_performance_json_parser.py, if you are adding new
           metrics the parser will need to be updated in order to correctly read in the
           new data.

        d. It must be able to create the figures displaying performance. This is also
           handled by a separate script the parthenon_performance_plotter.py, it may be
           approprite to adjust it as needed for new functionality or add a separate
           plotting script.

A few closing points, the advection performance tests are currently setup so that
if a user creates a pr to another branch the figures generated will contain the
performance metrics of the two branches and plot them. In the case that the performance
metrics are not being run as part of a pr, for example the performance of the development
branch is checked on a daily basis then the performance from the last 5 commits that
branches metrics file (stored on the wiki, performance_metrics_develop.json) will be plotted.

#### Including Tests in Code Coverage Report

Unit tests that are not performance tests (Do not have the [performance] tag) are
automatically included in the coverage report. If you want to include a performance
unit test in the coverage report the tag [coverage] should be added to the catch2
`TEST_CASE` macro. E.g.

```
TEST_CASE("test description", "[unit][performance][coverage]"){
...
}
```

It is advisable not to add the coverage tag to performance tests especially if they
take a long time to run, as adding code coverage will make them even slower. It is
better to reserve coverage for tests that show correct behavior.

Unlike unit tests, regression tests are not automatically added to the coverage report.
To add a regression test to the coverage report an attribute of the parameters argument
can be altered. E.g.

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
