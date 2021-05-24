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

import copy
import os
import datetime
import json
import numpy as np


class PerformanceDataJsonParser():
    """
    Performance Data Parser

    This class is responsible for reading performance data from json files. Including
    metadata associated with each test.
    """

    @staticmethod
    def _containsCommit(json_objs, value):
        """Will determine if a commit is found in the performance json files."""
        if not isinstance(json_objs, list):
            json_objs = [json_objs]

        for json_obj in json_objs:
            if json_obj.get('commit sha') == value:
                return True

        return False

    @staticmethod
    def _add_mesh_blocks_and_zone_cycles(obj1, obj2):
        """
        Add data from obj2 to obj1

        If the commits and tests match simply overwrite the metrics. If not
        tests match but the commit matches add the test and its metrics to the
        existing commit. If there are no matching commits do nothing return false.
        """
        if obj1.get('commit sha') == obj2.get('commit sha'):
            for data_grp1 in obj1.get('data'):
                for data_grp2 in obj2.get('data'):
                    if data_grp1.get('test') == data_grp2.get('test'):
                        # Overwrite the existing content with the new content
                        # If commit and test already exist
                        data_grp1['mesh_blocks'] = copy.deepcopy(
                            data_grp2.get('mesh_blocks'))
                        data_grp1['zone_cycles'] = copy.deepcopy(
                            data_grp2.get('zone_cycles'))
                        return True
            else:
                # If the commit matches but no matching tests are found
                # Add the data to the existing commit
                # Then the test was not found so we are going to append to it
                obj1['data'].extend(obj2['data'])
                return True
        return False

    def _add_to_json_obj(self, new_data):
        """
        json data should be of the following form:

        Where there are blocks of data in a list with the content below

                'commit sha': commit_sha,
                'branch': current_branch,
                'date': now.strftime("%Y-%m-%d %H:%M:%S"),
                'data':[{
                  'test': dir,
                  'mesh_blocks': mesh_blocks,
                  'zone_cycles': zone_cycles}]
        """
        # Ensure new_data is a block of data and not a list
        if isinstance(new_data, list):
            if len(new_data) == 1:
                new_data = new_data[0]
            else:
                raise ValueError("Expected exactly 1 new data")

        # Turn the class data into a list if it is not already one
        if not isinstance(self._data, list):
            self._data = [self._data]

        # Cycle the objects in the internal list, and if there is a match add the data
        for json_obj in self._data:
            if self._add_mesh_blocks_and_zone_cycles(json_obj, new_data):
                return

        # Cycle the outer list first
        if isinstance(self._data, list):
            for json_obj in self._data:
                if self._add_mesh_blocks_and_zone_cycles(json_obj, new_data):
                    return
        else:
            # If none of the commits match then add the whole of the new data block
            # Then the test was not found so we are going to append to it
            # Using append and not extend because new_data is gauranteed not to be a list
            self._data.append(new_data)

    @staticmethod
    def _getCyclesAndMeshblocks(json_obj, test):
        for data_grp in json_obj.get('data'):
            if data_grp.get('test') == test:
                mesh_blocks = data_grp.get('mesh_blocks')
                cycles = data_grp.get('zone_cycles')
        return mesh_blocks, cycles

    @staticmethod
    def _containsTest(json_obj, test):
        for data_grp in json_obj.get('data'):
            if data_grp.get('test') == test:
                return True
        return False


    def _getMeshBlocksOrCyclesAt(self, meshblock_or_cycles, commit_index, test):
        list_ind = 0
        for json_obj in self._data:
            if commit_index == list_ind:
                for data_grp in json_obj.get('data'):
                    if data_grp.get('test') == test:
                        if meshblock_or_cycles == "cycles":
                            cycles = data_grp.get('zone_cycles')
                            if isinstance(cycles, str):
                                cycles = np.array(
                                    cycles.strip("[").strip("]").split()).astype(
                                    np.float)
                            return cycles
                        else:
                            mesh_blocks = data_grp.get('mesh_blocks')
                            if isinstance(mesh_blocks, str):
                                mesh_blocks = np.array(
                                    mesh_blocks.strip("[").strip("]").split()).astype(
                                    np.float)
                            return mesh_blocks
            list_ind = list_ind + 1
        return None

    @staticmethod
    def getData(file_name):
        """Will read data from a performance file into json formated objected."""
        if os.path.isfile(file_name):
            # If does exist:
            # 1. load the
            if os.stat(file_name).st_size != 0:
                with open(file_name, 'r') as fid:
                    return json.load(fid)
        return None

    def getMostRecentPerformanceData(self, file_name, branch, test):
        """
        Will parse a performance .json file and get the latest metrics.

        file_name - file where the data is stored
        branch - is the branch we are trying to get performance metrics for
        test - is the test we are getting the metrics for

        Will return the mesh_blocks and cycles if found else will return None
        """
        if os.path.isfile(file_name):
            # If does exist:
            # 1. load the
            if os.stat(file_name).st_size != 0:
                with open(file_name, 'r') as fid:
                    json_objs = json.load(fid)

                    mesh_blocks = None
                    cycles = None

                    recent_datetime = None
                    if not isinstance(json_objs, list):
                        json_objs = [json_objs]

                    for json_obj in json_objs:
                        new_datetime = datetime.datetime.strptime(
                            json_obj.get('date'), '%Y-%m-%d %H:%M:%S')
                        if recent_datetime is None:
                            recent_datetime = new_datetime
                            if self._containsTest(json_obj, test):
                                mesh_blocks, cycles = self._getCyclesAndMeshblocks(
                                    json_obj, test)

                        if new_datetime > recent_datetime:
                            if self._containsTest(json_obj, test):
                                recent_datetime = new_datetime
                                mesh_blocks, cycles = self._getCyclesAndMeshblocks(
                                    json_obj, test)

                if isinstance(mesh_blocks, str):
                    mesh_blocks = np.array(
                        mesh_blocks.strip("[").strip("]").split()).astype(
                        np.float)
                if isinstance(cycles, str):
                    cycles = np.array(
                        cycles.strip("[").strip("]").split()).astype(
                        np.float)
                return mesh_blocks, cycles
        return None

    def append(self, new_data, file_name):
        """
        Append new data to a performance .json file.

        Will overwrite old data if the commit already exists in the file.
        """
        data_found = False
        if os.path.isfile(file_name):
            # If does exist:
            # 1. load the
            if os.stat(file_name).st_size != 0:
                with open(file_name, 'r') as fid:
                    data_found = True
                    # self._data will be a dict
                    self._data = json.load(fid)
                    if not isinstance(self._data,list):
                        self._data = [self._data]

                # Check if the commit exists in the data already
                if self._containsCommit(self._data, new_data['commit sha']):
                    self._add_to_json_obj(new_data)
                else:
                    self._data.append(new_data)

        if not data_found:
            self._data = new_data

        with open(file_name, 'w') as fout:
            # Need to convert the dict to a string to dump to a file
            json.dump(self._data, fout, indent=4)

    def getNumOfCommits(self):
        return len(self._data)

    def getCyclesAt(self, commit_index, test):
        """Returns the number of cycles for a particular test associated with a commit."""
        return self._getMeshBlocksOrCyclesAt("cycles", commit_index, test)

    def getMeshBlocksAt(self, commit_index, test):
        """Returns the number of mesh blocks for a particular test associated with a commit."""
        return self._getMeshBlocksOrCyclesAt("mesh_blocks", commit_index, test)

    def getCommitShaAt(self, commit_index, test):
        list_ind = 0
        for json_obj in self._data:
            if commit_index == list_ind:
                for data_grp in json_obj.get('data'):
                    if data_grp.get('test') == test:
                        return json_obj.get('commit sha')
            list_ind = list_ind + 1
        return None

    @staticmethod
    def checkDataUpToDate(file_name, branch, commit_sha, test):
        """Checks to see if performance metrics exist for the commit and test specified."""
        if not os.path.isfile(file_name):
            return False
        if os.stat(file_name).st_size == 0:
            return False
        with open(file_name, 'r') as fid:
            json_objs = json.load(fid)

            for json_obj in json_objs:
                if json_obj.get('commit sha') == commit_sha:
                    if json_obj.get('branch') == branch:
                        for data_grp in json_obj.get('data'):
                            if data_grp.get('test') == test:
                                return True

        return False
