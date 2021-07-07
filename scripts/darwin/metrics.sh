#!/bin/bash
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


# Load system env only
SCRIPT=$(realpath "$0")
SOURCE=$(dirname "$SCRIPT")
source "${SOURCE}/setup.sh"

if [[ "develop" == "${CI_COMMIT_BRANCH}" ]]; then
  # The CI runs on a schedule on the develop branch of parthenon, in this case there is
  # no target branch (branch to merge into) because the ci is not running for a pull
  # request. As such the target branch is equivalent to the feature branch which is "develop".
  target_branch="${CI_COMMIT_BRANCH}"
else

  # Here the parthenon metrics app is being used to communicate with the github server. It is
  # asking the github server what the target branch is. Note the parthenon metrics app is
  # not interacting locally with the github repository but is interacting with the remote
  # server using RESTful api calls, this ensures that the information it obtains is consistent
  # with the latest changes on the server.
  #
  # CI_COMMIT_BRANCH
  #
  # This is your current branch (feature branch).
  #
  # GITHUB_APP_PEM
  #
  # This is the path to the permissions file of the github application. The pem file is what
  # allows the application to interact with the parthenon github repository.
  data=$("${METRICS_APP}" -p "${GITHUB_APP_PEM}" --get-target-branch --branch "${CI_COMMIT_BRANCH}")

  echo "Get target branch or pr"
  echo "${data}"

  target_branch=$(echo "$data" | grep "Target branch is:" | awk '{print $4}')
fi

# Pathenon Metrics Application Analysis
#
# ********* Overview ********
#
# The below command at the bottome of the page s rather involved. Here the parthenon
# metrics app is being told to analyze the results of the parthenon performance
# regression tests. This consists of several steps:
#
# 1. performance test data must exist in the local parthenon repository
#
# 2. A comparison between the local test results and previous results must be conducted
#    so that a regression can be spotted.
#
#      * In the case of a pull request, this means comparing the local results with results
#        of the target branch.
#
#      * In the case of a scheduled run, where there is no merge, the local test results
#        which are assumed to be the most up to date are compared with the performance of
#        previous commits on the same branch.
#
#    The performance metrics of the non local repository are stored in the parthenon wiki.
#    The metrics app will download the wiki and read the performance metrics from the
#    appropriate files in the wiki to make the comparison.
#
# 3. Then the metrics app will generate figures displaying performance data (regressions should be easy to spot).
#
# 4. Next a wiki page file will be generated in the wiki with links to the figures. This
#    wiki page file will then be uploaded to the parthenon wiki.
#
# 5. The figures will be uploaded to an empty figures branch (orphan branch) in the parthenon repository:
#    https://github.com/lanl/parthenon/tree/figures
#
#    It may be necessary in the future to squash the commit history in that branch occassionally.
#
# 6. Finally, a status report will be sent to github and associated with commit of the current branch,
#    this status should show up next to other status reports such as Codacy, Deepcode and gitlab ci.
#    The status will contain a link to the wiki page where the performance metrics are posted.
#
# NOTE a. Why are we uploading figures to their own branch in the parthenon repository and not
#      simply storing them in parthenon.wiki?
#
#      Binary files will bloat the wiki repository
#
#      b. Why not simply place the image files in a branch on wiki?
#
#      I could not figure out how to link to the images that are located in a branch
#      in the wiki repository. from the wiki page.
#
#      c. Why not store them in develop in the partheon repo?
#
#      Again we don't want to bloat the partheon repo with binary files.
#
# ********** Flags and thier meanings **********
#
# -p path_to_matrics_application_permissions_file
#
# The path to the .pem file so the metrics application can authenticate with the github server.
#
# --analyze path_to_tst_results
#
# In order to analyze, the results of the tests of the local repository the metrics application needs to
# know where performance test output is located. So for instance if you build parthenon with the command:
#
# cmake -S. -B build
# cmake --build build
#
# and then run the tests
#
# cmake --build build --target tests
#
# Then the tst output will appear in /build/tst/regression/outputs this is what is being
# passed to the --analyze flag
#
# --create
#
# This flag is simply to allow the metrics app to create branches on the wiki or parthenon repository if
# needed. So for instance if the figures branch in parthenon repository was deleted for some reason
# the metrics app would be able to recreate it.
#
# --post-analyze-status
#
# This flag indicates that the metrics app will post a status to the github repository once it is done analyzing.
# If this flag was not set you could still post a status but it would require another call to the metrics app.
#
# --branch current_branch_name
#
# This is the branch that the local performance tests were run for.
#
# --target_branch target_branch_name
#
# This is the branch that is being merged into if it is a pull request.
#
# --generate-figures-on-analysis
#
# Does what it says.
"${METRICS_APP}" -p "${GITHUB_APP_PEM}" --analyze "${BUILD_DIR}/tst/regression/outputs" --create --post-analyze-status --branch "${CI_COMMIT_BRANCH}" --target-branch "$target_branch" --generate-figures-on-analysis
