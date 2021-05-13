#!/usr/bin/env python3
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

import logging
import matplotlib.pyplot as plt

class PerformanceMetricsPlotter():

    def __init__(self,
                 number_commits_to_plot,
                 test_dir,
                 current_branch,
                 mesh_blocks,
                 zone_cycles,
                 target_branch,
                 target_data_file_exists,
                 target_meshblocks,
                 target_cycles):
        """Creates figures that display performance metrics."""
        self._number_commits_to_plot = number_commits_to_plot
        self._test_dir = test_dir
        self._current_branch = current_branch
        self._mesh_blocks = mesh_blocks
        self._zone_cycles = zone_cycles
        self._target_branch = target_branch
        self._target_data_file_exists = target_data_file_exists
        self._target_meshblocks = target_meshblocks
        self._target_cycles = target_cycles

        self._log = logging.getLogger("performance_plotter")
        self._log.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        self._log.addHandler(ch)

    def _plotDataFromPreviousCommitsFromSameBranch(self, json_perf_data_parser, figure_path_name):
        # If running on same branch grab the data for the last
        # 5 commits stored in the file
        fig, p = plt.subplots(
            2, 1, figsize=(4, 8), sharex=True)
        legend_temp = []
        for i in range(0, self._number_commits_to_plot):
            mesh_blocks_temp = json_perf_data_parser.getMeshBlocksAt(
                i, self._test_dir)
            cycles_temp = json_perf_data_parser.getCyclesAt(
                i, self._test_dir)
            commit_temp = json_perf_data_parser.getCommitShaAt(
                i, self._test_dir)
            if mesh_blocks_temp is None:
                self._log.info(
                    "Skipping data at %s\n no mesh blocks recordered in data" %
                    commit_temp)
                continue
            if cycles_temp is None:
                self._log.info(
                    "Skipping data at %s\n no cycles recordered in data" %
                    commit_temp)
                continue

            if i == 0:
                norm_const = cycles_temp[0]
            p[0].loglog(
                mesh_blocks_temp, cycles_temp, label="$256^3$ Mesh")
            p[1].loglog(
                mesh_blocks_temp, norm_const / cycles_temp)
            legend_temp.append(commit_temp)

        for i in range(2):
            p[i].grid()

        p[0].legend(legend_temp)
        p[1].legend(legend_temp)

        p[0].set_ylabel("zone-cycles/s")
        p[1].set_ylabel("normalized overhead")
        p[1].set_xlabel("Meshblock size")

        fig.suptitle(self._test_dir, fontsize=16)
        fig.savefig(figure_path_name, bbox_inches='tight')

    def _plotTargetBranchDataVsCurrentBranchData(self, figure_path_name):
        # Get the data for the last commit in the development branch
        # Now we need to create the figure to update
        fig, p = plt.subplots(
            2, 1, figsize=(4, 8), sharex=True)

        p[0].loglog(
            self._mesh_blocks, self._zone_cycles, label="$256^3$ Mesh")
        p[1].loglog(self._mesh_blocks,
                    self._zone_cycles[0] / self._zone_cycles)
        if self._target_data_file_exists:
            p[0].loglog(
                self._target_meshblocks,
                self._target_cycles,
                label="$256^3$ Mesh")
            p[1].loglog(
                self._target_meshblocks, self._zone_cycles[0] / self._target_cycles)

        for i in range(2):
            p[i].grid()

        if self._target_data_file_exists:
            p[0].legend([self._current_branch, self._target_branch])
            p[1].legend([self._current_branch, self._target_branch])
        else:
            p[0].legend([self._current_branch])
            p[1].legend([self._current_branch])
        p[0].set_ylabel("zone-cycles/s")
        p[1].set_ylabel("normalized overhead")
        p[1].set_xlabel("Meshblock size")
        fig.savefig(figure_path_name, bbox_inches='tight')

    def plot(self, json_perf_data_parser, figure_path):
        """
        Create a plot of the metrics

        The figure_path is where the figure will be saved too, if the target branch is the same as
        the current branch then metrics associated with previous commits from the same branch will
        be plotted. If the target and current branch are different then the latest commits from
        both will be plotted.
        """
        if self._target_branch == self._current_branch:
            self._plotDataFromPreviousCommitsFromSameBranch(
                json_perf_data_parser, figure_path)
        else:
            self._plotTargetBranchDataVsCurrentBranchData(figure_path)
