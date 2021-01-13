#!/usr/bin/env python3
#=========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

import argparse
import os
import jwt
import pem
import datetime
import pathlib
import pprint
import pycurl
import json
import shutil
import base64
from io import BytesIO
from git import Repo
from app import App


class PerformanceDataJsonParser():

  def __containsCommit(self, json_objs, value):
    if not isinstance(json_objs, list):
      json_objs = [json_objs]

    for json_obj in json_objs:
      if json_obj.get('commit sha') == value:
        return True

    return False

  def __add_to_json_obj(self,new_data):
    """ json data should be of the following form:

        Where there are blocks of data in a list with the content below

                'commit sha': commit_sha, 
                'branch': current_branch,
                'date': now.strftime("%Y-%m-%d %H:%M:%S"),
                'data':[{
                  'test': dir,
                  'mesh_blocks': mesh_blocks,
                  'zone_cycles': zone_cycles}]
    """
    # Cycle the outer list first
    for json_obj in self.__data:
      if json_obj.get('commit sha') == new_data.get('commit sha'):
        for data_grp in json_obj.get('data'):
          if data_grp.get('test') == new_data.get('test'):
            # Overwrite the existing content with the new content
            data_grp['mesh_blocks'] = new_data.get('mesh_blocks')
            data_grp['zone_cycles'] = new_data.get('zone_cycles')
            return
        # Then the test was not found so we are going to append to it
        json_obj['data'].append(new_data['data'])

  def getData(self, file_name):
    if os.path.isfile(file_name):
      # If does exist:
      # 1. load the 
      with open(file_name, 'r') as fid:
        return json.loads(fid)

  def getMostRecentPerformanceData(self, file_name, branch, test):
    if os.path.isfile(file_name):
      # If does exist:
      # 1. load the 
      with open(file_name, 'r') as fid:
        json_objs = json.loads(fid)

        mesh_blocks = None
        cycles = None

        recent_datetime = None
        for json_obj in json_objs:
          new_datetime = datetime.datetime.strptime(json_obj.get('date'), '%Y-%m-%d %H:%M:%S')
          if recent_datetime == None:
            recent_datetime = new_datetime
            for data_grp in json_obj.get('data'):
              if data_grp.get('test') == test:
                mesh_blocks = data_grp.get('mesh_blocks')
                cycles = data_grp.get('zone_cycles')

          if new_datetime > recent_datetime:
            recent_datetime = new_datetime
            for data_grp in json_obj.get('data'):
              if data_grp.get('test') == test:
                mesh_blocks = data_grp.get('mesh_blocks')
                cycles = data_grp.get('zone_cycles')
      return mesh_blocks, cycles

  def append(self, new_data, file_name):

    if os.path.isfile(file_name):
      # If does exist:
      # 1. load the 
      with open(file_name, 'r') as fid:
        self.__data = json.loads(fid)

      # Check if the commit exists in the data already
      if self.__containsCommit(self.__data, new_data['commit sha']):
        # Should really cycle through the list and see if the test already exists if it does it
        # should overwrite it
        # TODO
        self.__add_to_json_obj(new_data)
      else:
        self.__data.update(new_data) 
    else:
      self.__data = new_data

    with open(json_file_out, 'w') as fout:
      json_dumps_str = json.dumps(self.__data, indent=4)
      print(json_dumps_str, file=fout)


"""
Parthenon App Class

This class is responsible for authenticating against the parthenon repository and interacting
with the github api. 
"""
class ParthenonApp(App):

  """
  Internal Private Methods
  """
  def __init__(self):
    super().__init__(
        92734,
        "Parthenon_Github_Metrics_Application",
        "lanl",
        "parthenon")

  def initialize(self,use_wiki=False, ignore=False, pem_file = "", create_branch=False):
    super().initialize(use_wiki, ignore, pem_file, create_branch)

  def readPerformanceMetricsTXT(self,file_path):
    mesh_blocks = np.zeros()
    zone_cycles = np.zeros()
    with open(file_path,'r') as reader:
      lines = reader.readlines() 
      # Remove first line in file, it is just the title
      lines.pop()

      mesh_block = np.resize(mesh_block, len(lines))
      zone_cycles = np.resize(zone_cycles, len(lines))

      ind = 0
      for line in lines:
        line = line.split()
        mesh_block[ind] = lines[2]
        zone_cycles[ind] = lines[0]
        ind = ind + 1
    return mesh_blocks, zone_cycles

  def analyze(self, regression_outputs):
    regression_outputs = os.path.abspath(regression_outputs)
    if not os.path.exists(regression_outputs):
      raise Exception("Cannot analyze regression outputs specified path is invalid")
    if not os.path.isdir(regression_outputs):
      raise Exception("Cannot analyze regression outputs specified path is invalid")
    
    current_branch = os.getenv('CI_COMMIT_BRANCH')
    target_branch = super().getBranchMergingWith(current_branch)
    pr_wiki_page = os.path.join(str(self.__parthenon_wiki_dir), branch + "_" + target_branch + ".md" )

    all_dirs = os.listdir(regression_outputs)
    print("Contents of regression_outputs: %s" % regression_outputs )
    for test_dir in all_dirs:
      print(test_dir)
      if test_dir == "advection_performance":
        if not os.path.isfile(regression_outputs + "/advection_performance/performance_metrics.txt"):
          raise Exception("Cannot analyze advection_performance, missing performance metrics file.")
        repo = super().cloneWikiRepo()

        mesh_blocks, zone_cycles = self.readPerformanceMetricsTXT(regression_outputs + "/advection_performance/performance_metrics.txt")
        now = datetime.datetime.now()
        
        # Check if performance_metrics.json exists in wiki
        # It actually makes the most sense to store each performance metric in it's own file to 
        # avoid merge conflicts. 
        # The content of each file should contain the commit
        # The date
        # The performance metrics 
        # The pull request
        commit_sha = os.getenv('CI_COMMIT_SHA')
        new_data = {
            'commit sha': commit_sha, 
            'branch': current_branch,
            'date': now.strftime("%Y-%m-%d %H:%M:%S"),
            'data':[{
            'test': dir,
            'mesh_blocks': mesh_blocks,
            'zone_cycles': zone_cycles
                  }]
            }

        json_file_out = str(self.__parthenon_wiki_dir) + "/performance_metrics_"+ current_branch + ".json"
        json_perf_data_parser = PerformanceDataJsonParser()
        json_perf_data_parser.append(new_data, json_file_out)
     
        # Now the new file needs to be committed
        upload(json_file_out, "master",use_wiki=True)

        json_file_compare = str(self.__parthenon_wiki_dir) + "/performance_metrics_" + target_branch + ".json"
        
        target_data_file_exists = False
        if os.path.isfile(json_file_compare):
          target_data_file_exists = True
          target_meshblocks, target_cycles = json_perf_data_parser.getMostRecentPerformanceData(json_file_compare, target_branch, test_dir)
        # Get the data for the last commit in the development branch

        # Now we need to create the figure to update
        fig, p = plt.subplots(2, 1, figsize = (4,8), sharex=True)

        p[0].loglog(mesh_blocks, zone_cycles, label = "$256^3$ Mesh")
        p[1].loglog(mesh_blocks, zone_cycles[0]/zone_cycles)
        if target_data_file_exists:
          p[0].loglog(target_meshblocks, target_cycles, label = "$256^3$ Mesh")
          p[1].loglog(target_mesh_blocks, zone_cycles[0]/target_cycles)

        for i in range(2):
            p[i].grid()

        if target_data_file_exists:
          p[0].legend([branch,target_branch])
          p[1].legend([branch,target_branch])
        else:
          p[0].legend([branch])
          p[0].legend([branch,target_branch])
        p[0].set_ylabel("zone-cycles/s")
        p[1].set_ylabel("normalized overhead")
        p[1].set_xlabel("Meshblock size")
        figure_name =test_dir + "_" + branch + "_" + target_branch + ".png"
        figure_path_name = os.path.join(str(self.__parthenon_wiki_dir), figure_name )
        fig.savefig(figure_path_name, bbox_inches='tight')
        upload(figure_path_name, "master",use_wiki=True)

        fig_url ='https://github.com/' + self.__user + '/' + self.__repo_name + '/blob/figures/' + figure_name + '?raw=true'
      
      elif test_dir == "advection_performance_mpi":
        if not os.path.isfile(regression_outputs + "/advection_performance_mpi/performance_metrics.txt"):
          raise Exception("Cannot analyze advection_performance_mpi, missing performance metrics file.")

      # Check that the wiki exists for merging between these two branches, only want a single wiki page per merge

      with open(pr_wiki_page,'w') as writer: 
        writer.write("This file is managed by the " + self.__name + "\n")
        writer.write("![Image](" + fig_url +")\n")
    # 1 search for files 
    # 2 load performance metrics from wiki
    # 3 compare the metrics
    # 4 Create figure
    # 5 upload figure
    # 6 indicate pass or fail with link to figure


def main(**kwargs):

  app = ParthenonApp()
  app.initialize(
      kwargs.pop('wiki'),
      kwargs.pop('ignore'),
      kwargs.pop('permissions'),
      kwargs.pop('create'))

  branch = kwargs.pop('branch')

  if isinstance(branch,list):
    branch = branch[0]

  if 'upload' in kwargs:
    value = kwargs.pop('upload')
    if isinstance(value,list):
      value = value[0]
    if value != None:
        app.upload(value, branch)

  if 'status' in kwargs:
    value = kwargs.pop('status')
    if isinstance(value,list):
        value = value[0]
    if value != None:
        app.postStatus(value)

  if 'analyze' in kwargs:
    value = kwargs.pop('analyze')
    if isinstance(value,list):
        value = value[0]
    if value != None:
        app.analyze(value)

# Execute main function
if __name__ == '__main__':

    parser = argparse.ArgumentParser("python3 parthenon_metrics_app.py -p file.pem")
    
    desc = ('Path to the (permissions file/permissions string) which authenticates the application. If not provided will use the env variable PARTHENON_METRICS_APP_PEM.')
   
    parser.add_argument('--permissions','-p',
                        type=str,
                        nargs=1,
                        required=False,
                        help=desc)

    desc = ('Path to file want to upload.')
    parser.add_argument('--upload','-u',
                        type=str,
                        nargs=1,
                        required=False,
                        help=desc)


    desc = ('Branch to use. Default is develop')
    parser.add_argument('--branch','-b',
                        type=str,
                        nargs=1,
                        required=False,
                        default = "develop",
                        help=desc)

    desc = ('Post current status state: error, failed, pending or success.')
    parser.add_argument('--status','-s',
            type=str,
            nargs=1,
            required=False,
            help=desc)

    desc = ('Path to regression tests output, to analyze.')
    parser.add_argument('--analyze','-a',
        type=str,
        nargs=1,
        required=False,
        help=desc)

    desc = ('Create Branch if does not exist.')
    parser.add_argument('--create','-c',
        action='store_true',
        default=False,
        help=desc)

    desc = ('Use the wiki repository.')
    parser.add_argument('--wiki','-w',
        action='store_true',
        default=False,
        help=desc)

    desc = ('Ignore rules, will ignore upload rules')
    parser.add_argument('--ignore','-i',
        action='store_true',
        default=True,
        help=desc)

    args = parser.parse_args()
    try:
        main(**vars(args))
    except Exception:
        raise


