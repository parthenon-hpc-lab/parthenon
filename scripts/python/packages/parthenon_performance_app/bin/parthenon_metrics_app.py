#!/projects/parthenon-int/parthenon-project/views/darwin/ppc64le/gcc9/2021-04-08/bin/python3.9
# =========================================================================================
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
# =========================================================================================

import argparse
import os
import datetime
import parthenon_performance_app.githubapp
from parthenon_performance_app.parthenon_performance_advection_analyzer import (
    AdvectionAnalyser,
)
from parthenon_performance_app.parthenon_performance_json_parser import (
    PerformanceDataJsonParser,
)


class ParthenonApp(parthenon_performance_app.githubapp.GitHubApp):
    """
    Parthenon App Class

    This class is responsible for authenticating against the parthenon repository and interacting
    with the github api.
    """

    def __init__(self):
        """
        The parthenon metrics app is initialized with the following arguments:
        * github app id
        * the name of the application
        * the owner of the repository it has access to
        * the name of the repository
        """
        super().__init__(
            92734,
            "Parthenon_Github_Metrics_Application",
            "lanl",
            "parthenon",
            os.path.dirname(os.path.realpath(__file__)),
        )

    def _createFigureURLPathAndName(self, test_dir, current_branch, target_branch):
        figure_name = (
            test_dir
            + "_"
            + current_branch.replace(r"/", "-")
            + "_"
            + target_branch.replace(r"/", "-")
            + ".png"
        )
        figure_path_name = os.path.join(self._parthenon_wiki_dir, figure_name)
        fig_url = (
            "https://github.com/"
            + self._user
            + "/"
            + self._repo_name
            + "/blob/figures/"
            + figure_name
            + "?raw=true"
        )
        self._log.info("Figure url is: %s" % fig_url)
        return fig_url, figure_path_name, figure_name

    def _writeWikiPage(
        self, commit_sha, pr_wiki_page, figure_urls, now, wiki_file_name
    ):
        """Write the contents of the performance metrics into a wiki page."""
        with open(pr_wiki_page, "w") as writer:
            writer.write("This file is managed by the " + self._name + ".\n\n")
            writer.write("Date and Time: %s\n" % now.strftime("%Y-%m-%d %H:%M:%S"))
            writer.write("Commit: %s\n\n" % commit_sha)
            for figure_url in figure_urls:
                writer.write("![Image]({})\n\n".format(figure_url))
            wiki_url = "https://github.com/{usr_name}/{repo_name}/wiki/{file_name}"
            wiki_url = wiki_url.format(
                usr_name=self._user, repo_name=self._repo_name, file_name=wiki_file_name
            )
            self._log.info("Wiki page url is: %s" % wiki_url)
            return wiki_url

    def _uploadMetrics(self, json_files_to_upload, figure_files_to_upload, pr_wiki_page):
        """
        Uploads metrics files

        Metrics files include:
        json data files containing raw data
        figure files containging plots of the performance metrics
        the wiki page used to display the metrics on github
        """
        for json_upload in json_files_to_upload:
            self.upload(json_upload, "master", use_wiki=True, wiki_state="mixed")

        for figure_upload in figure_files_to_upload:
            self.upload(
                figure_upload,
                self._default_image_branch,
                use_wiki=False,
                wiki_state="mixed",
            )

        # wiki_state soft prevents overwriting changes made to files that exist
        # within the repo
        self.upload(pr_wiki_page, "master", use_wiki=True, wiki_state="mixed")

    def getCurrentAndTargetBranch(self, branch):
        """
        Returns the branch that the current branch and the branch that is being merged with (the target branch).

        If no pull request is open returns None for the target.
        """
        target_branch = super().getBranchMergingWith(branch)
        self._log.info(
            "Current branch: %s\nTarget Branch: %s" % (branch, target_branch)
        )
        return branch, target_branch

    def analyze(
        self,
        regression_outputs,
        current_branch,
        target_branch,
        post_status,
        create_figures,
        number_commits_to_plot=5,
    ):
        """
        This method will analayze the output of test performance regression metrics

        The output from each test will be used to create a figure demonstrating the
        performance. The output will then be recorded in the json formatted performance
        metrics file stored in the wiki. Each branch has it's own performance metrics file.
        The performance metrics from the latest commit of the target branch (the branch to
        be merged into) are read in and plotted alongside the current metrics.
        The figures depecting the performance are then uploaded to a orphan branch named figures.
        Links to the figures are created in a markdown file stored on the wiki which is
        also uploaded.

        Finally, a status check is posted to the pr with a link to the performance metrics.
        """
        regression_outputs = os.path.abspath(regression_outputs)
        if not os.path.exists(regression_outputs):
            raise Exception(
                "Cannot analyze regression outputs specified path is invalid: "
                + regression_outputs
            )
        if not os.path.isdir(regression_outputs):
            raise Exception(
                "Cannot analyze regression outputs specified path is invalid: "
                + regression_outputs
            )

        if isinstance(current_branch, list):
            current_branch = current_branch[0]
        if isinstance(target_branch, list):
            target_branch = target_branch[0]

        wiki_file_name = (
            current_branch.replace(r"/", "-") + "_" + target_branch.replace(r"/", "-")
        )
        pr_wiki_page = os.path.join(self._parthenon_wiki_dir, wiki_file_name + ".md")

        all_dirs = os.listdir(regression_outputs)
        self._log.info("Contents of regression_outputs: %s" % regression_outputs)

        # Make sure wiki exists
        super().cloneWikiRepo()

        now = datetime.datetime.now()

        commit_sha = os.getenv("CI_COMMIT_SHA")
        # Files should only be uploaded if there are no errors
        json_files_to_upload = []
        figure_files_to_upload = []
        figure_urls = []
        for test_dir in all_dirs:
            if not isinstance(test_dir, str):
                test_dir = str(test_dir)
            if (
                test_dir == "advection_performance"
                or test_dir == "advection_performance_mpi"
            ):

                figure_url, figure_file, _ = self._createFigureURLPathAndName(
                    test_dir, current_branch, target_branch
                )

                analyzer = AdvectionAnalyser(create_figures)
                json_file_out = analyzer.analyse(
                    regression_outputs,
                    commit_sha,
                    test_dir,
                    target_branch,
                    current_branch,
                    self._parthenon_wiki_dir,
                    figure_file,
                    number_commits_to_plot,
                    now,
                )

                json_files_to_upload.append(json_file_out)

                if create_figures:
                    figure_files_to_upload.append(figure_file)
                    figure_urls.append(figure_url)

        wiki_url = self._writeWikiPage(
            commit_sha, pr_wiki_page, figure_urls, now, wiki_file_name
        )

        self._uploadMetrics(json_files_to_upload, figure_files_to_upload, pr_wiki_page)

        if post_status:
            self.postStatus(
                "success",
                commit_sha,
                context="Parthenon Metrics App",
                description="Performance Regression Analyzed",
                target_url=wiki_url,
            )
        # 1 search for files
        # 2 load performance metrics from wiki
        # 3 compare the metrics
        # 4 Create figure
        # 5 upload figure
        # 6 indicate pass or fail with link to figure

    def checkUpToDate(self, target_branch, tests):
        """Check to see if performance metrics for all the tests exist."""
        super().cloneWikiRepo()
        target_file = (
            str(self._parthenon_wiki_dir)
            + "/performance_metrics_"
            + target_branch.replace(r"/", "-")
            + ".json"
        )
        isUpToDate = True
        if os.path.isfile(target_file):
            if self.branchExist(target_branch):
                json_perf_data_parser = PerformanceDataJsonParser()
                commit_sha = self.getLatestCommitSha(target_branch)
                for test in tests:
                    test_isUpToDate = json_perf_data_parser.checkDataUpToDate(
                        target_file, target_branch, commit_sha, test
                    )
                    self._log.info(
                        "Performance Metrics for test %s is uptodate: %s"
                        % test_isUpToDate
                    )
                    if not test_isUpToDate:
                        isUpToDate = False
            else:
                self._log.warning(
                    "Branch (%s) is not available on github." % target_branch
                )
                isUpToDate = False
        else:
            isUpToDate = False
            self._log.warning("Performance Metrics file is missing.")

        self._log.info("Performance Metrics are uptodate: %s" % isUpToDate)

    def printTargetBranch(self, branch):
        target_branch = self.getBranchMergingWith(branch)
        if target_branch is None:
            self._log.warning(
                "Branch (%s) does not appear to not have an open pull request, no target detected."
                % branch
            )
        else:
            self._log.info("Target branch is: %s" % target_branch)


def getValue(kwargs, name):
    value = kwargs[name]
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        else:
            raise ValueError(
                "Expected exactly 1 value for %s, but got %i" % (name, len(value))
            )
    else:
        return value


def main(**kwargs):

    app = ParthenonApp()
    app.initialize(
        kwargs["wiki"],
        kwargs["ignore"],
        kwargs["permissions"],
        kwargs["create"],
        getValue(kwargs, "repository_path"),
    )

    branch = getValue(kwargs, "branch")

    if "upload" in kwargs:
        value = getValue(kwargs, "upload")
        if value is not None:
            app.upload(value, branch)

    if "status" in kwargs:
        value = getValue(kwargs, "status")
        if value is not None:
            url = getValue(kwargs, "status_url")
            context = getValue(kwargs, "status_context")
            description = getValue(kwargs, "status_description")
            app.postStatus(value, None, context, description, target_url=url)

    if "analyze" in kwargs:
        value = getValue(kwargs, "analyze")
        if value is not None:
            target_branch = getValue(kwargs, "target_branch")
            if target_branch == "":
                _, target_branch = app.getCurrentAndTargetBranch(branch)
                # If target branch is None, assume it's not a pull request
                if target_branch is None:
                    target_branch = branch
            app.analyze(
                value,
                branch,
                target_branch,
                getValue(kwargs, "post_analyze_status"),
                getValue(kwargs, "generate_figures_on_analysis"),
            )

    if getValue(kwargs, "check_branch_metrics_uptodate"):
        app.checkUpToDate(branch, kwargs["tests"])

    if getValue(kwargs, "get_target_branch"):
        app.printTargetBranch(branch)


# Execute main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser("python3 parthenon_metrics_app.py -p file.pem")

    desc = (
        "Path to the (permissions file/permissions string) which authenticates "
        "the application. If not provided will use the env variable PARTHENON_METRICS_APP_PEM."
    )

    parser.add_argument("--permissions", "-p", default="", type=str, nargs=1, help=desc)

    desc = (
        "Path to the repository that will be analized, by default will check to see if a "
        "repository was already specified."
    )

    parser.add_argument(
        "--repository-path", "-rp", default=None, type=str, nargs=1, help=desc
    )

    desc = "Path to file want to upload."
    parser.add_argument("--upload", "-u", type=str, nargs=1, help=desc)

    desc = "Branch to use. Default is develop"
    parser.add_argument(
        "--branch", "-b", type=str, nargs=1, default="develop", help=desc
    )

    desc = (
        "Target branch to use. Default is calculated by making a RESTful "
        "API to github using the branch pased in with --branch argument"
    )
    parser.add_argument(
        "--target-branch", "-tb", type=str, nargs=1, default="", help=desc
    )

    desc = "Post current status state: error, failed, pending or success."
    parser.add_argument("--status", "-s", type=str, nargs=1, help=desc)

    desc = "Post url to use with status."
    parser.add_argument("--status-url", "-su", default="", type=str, nargs=1, help=desc)

    desc = "Post description with status."
    parser.add_argument(
        "--status-description", "-sd", default="", type=str, nargs=1, help=desc
    )

    desc = "Post context to use with status."
    parser.add_argument(
        "--status-context", "-sc", default="", type=str, nargs=1, help=desc
    )

    desc = "Path to regression tests output, to analyze."
    parser.add_argument("--analyze", "-a", type=str, nargs=1, help=desc)

    desc = "Post analyze status on completion"
    parser.add_argument(
        "--post-analyze-status", "-pa", action="store_true", default=False, help=desc
    )

    desc = "Generate figures during analysis."
    parser.add_argument(
        "--generate-figures-on-analysis",
        "-gfoa",
        action="store_true",
        default=False,
        help=desc,
    )

    desc = "Create Branch if does not exist."
    parser.add_argument("--create", "-c", action="store_true", default=False, help=desc)

    desc = "Use the wiki repository."
    parser.add_argument("--wiki", "-w", action="store_true", default=False, help=desc)

    desc = (
        "Check if the performance metrics for the branch "
        'are uptodate, default branch is "develop"'
    )
    parser.add_argument(
        "--check-branch-metrics-uptodate",
        "-cbmu",
        action="store_true",
        default=False,
        help=desc,
    )

    desc = "Tests to analyze."
    parser.add_argument("--tests", "-t", nargs="+", default=[], type=str, help=desc)

    desc = "Ignore rules, will ignore upload rules"
    parser.add_argument("--ignore", "-i", action="store_true", default=True, help=desc)

    desc = "Get the target branch of the current pull request"
    parser.add_argument(
        "--get-target-branch", "-gtb", action="store_true", default=False, help=desc
    )

    args = parser.parse_args()

    main(**vars(args))
