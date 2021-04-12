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

import os
import datetime
import numpy as np
from parthenon_performance_app.parthenon_performance_json_parser import PerformanceDataJsonParser
from parthenon_performance_app.parthenon_performance_plotter import PerformanceMetricsPlotter


class AdvectionAnalyser():
    def __init__(self, create_figures):
        self._create_figures = create_figures

    def readPerformanceMetricsTXT(self, file_path):
        """Will read the performance metrics of a .txt file that is output from one of the tests"""
        mesh_blocks = np.zeros(1)
        zone_cycles = np.zeros(1)
        with open(file_path, 'r') as reader:
            lines = reader.readlines()
            # Remove first line in file, it is just the title

            mesh_blocks = np.resize(mesh_blocks, len(lines) - 1)
            zone_cycles = np.resize(zone_cycles, len(lines) - 1)

            ind = 0
            for line in lines:
              # Skip header
                if ind != 0:
                    line = line.split()
                    mesh_blocks[ind - 1] = float(line[2])
                    zone_cycles[ind - 1] = float(line[0])
                ind = ind + 1
        return mesh_blocks, zone_cycles

    def analyse(self,
                regression_outputs,
                commit_sha,
                test_dir,
                target_branch,
                current_branch,
                wiki_directory,
                figure_path_name,
                number_commits_to_plot,
                now
                ):

        if not os.path.isfile(
                regression_outputs + "/advection_performance/performance_metrics.txt"):
            raise Exception(
                "Cannot analyze advection_performance, missing performance metrics file: " +
                regression_outputs +
                "/advection_performance/performance_metrics.txt")
#        super().cloneWikiRepo()
        mesh_blocks, zone_cycles = self.readPerformanceMetricsTXT(
            regression_outputs + "/advection_performance/performance_metrics.txt")
        now = datetime.datetime.now()

        # Check if performance_metrics.json exists in wiki
        # It actually makes the most sense to store each performance metric in it's own file to
        # avoid merge conflicts.
        # The content of each file should contain the commit
        # The date
        # The performance metrics
        # The pull request
        new_data = {
            'commit sha': commit_sha,
            'branch': current_branch,
            'date': now.strftime("%Y-%m-%d %H:%M:%S"),
            'data': [{
                'test': test_dir,
                'mesh_blocks': np.array2string(mesh_blocks),
                'zone_cycles': np.array2string(zone_cycles)
            }]
        }

        # Get the data for the target branch before writing the stats for the current branch,
        # This is to avoid the scenario where the target and current
        # branch are the same.
        json_file_compare = str(
            wiki_directory) + "/performance_metrics_" + target_branch.replace(
            r'/', '-') + ".json"

        json_perf_data_parser = PerformanceDataJsonParser()
        target_data_file_exists = False
        if os.path.isfile(json_file_compare):
            target_data_file_exists = True
            target_meshblocks, target_cycles = json_perf_data_parser.getMostRecentPerformanceData(
                json_file_compare, target_branch, test_dir)

        json_file_out = str(
            wiki_directory) + "/performance_metrics_" + current_branch.replace(
            r'/', '-') + ".json"
        json_perf_data_parser.append(new_data, json_file_out)

        if self._create_figures:

            plotter = PerformanceMetricsPlotter(
                number_commits_to_plot,
                test_dir,
                current_branch,
                mesh_blocks,
                zone_cycles,
                target_branch,
                target_data_file_exists,
                target_meshblocks,
                target_cycles)

            plotter.plot(json_perf_data_parser, figure_path_name)
        return json_file_out
