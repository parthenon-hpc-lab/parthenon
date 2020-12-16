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
import pycurl
import json
import shutil
import base64
from io import BytesIO
from git import Repo

class Node:
  def __init__(self, dir_name = "", rel_path = ""):
    self.dir = dir_name
    self.dirs = []
    self.files = []
    self.misc = []
    self.rel_path = rel_path + dir_name

  def insert(self, content, content_type):
    if content_type == "dir":
      self.dirs.append(Node(content, self.rel_path + "/"))
    elif content_type == "file":
      self.files.append(content)
    else:
      self.files.append(content)

  def getNodes(self):
    return self.dirs

  def getPath(self):
    return self.rel_path

  def printTree(self):
    print("Contents in folder: " + self.rel_path)
    for fil in self.files:
      print("File " + fil)
    for mis in self.misc:
      print("Misc " + mis)
    for node in self.dirs:
      node.printTree()

class ParthenonApp:
  def __init__(self, use_wiki=False, ignore=False, pem_file = "", create_branch=False):

    self.__app_id = 92734
    self.__ignore = ignore
    self.__use_wiki = use_wiki
    self.__name = "Parthenon_Github_Metrics_Application"
    self.__user = "lanl"
    self.__repo_name = "parthenon"
    self.__repo_url = "https://api.github.com/repos/" + self.__user + "/" + self.__repo_name
    if isinstance(create_branch,list):
      self.__create_branch = create_branch[0]
    else:
      self.__create_branch = create_branch
    self.__default_branch = "develop"
    self.__default_image_branch = "figures"
    self.__branches = []
    self.__branch_current_commit_sha = {}
    self.__api_version = "application/vnd.github.v3+json"
    self.__parth_root = Node()
    self.__parthenon_home = str(pathlib.Path(__file__).parent.absolute())

    try:
      self.__parthenon_home = self.__parthenon_home[:self.__parthenon_home.index(self.__repo_name) + len("/" + self.__repo_name )] 
    except Exception:
      error_msg = str(os.path.realpath(__file__)) + " must be run from within the " + self.__repo_name + " repository."
      print(error_msg)
      raise

    self.__parthenon_wiki_dir = self.__parthenon_home +"/"+ self.__repo_name + ".wiki"
    if isinstance(pem_file,list):
      self.__generateJWT(pem_file[0])
    else:
      self.__generateJWT(pem_file)
    self.__generateInstallationId() 
    self.__generateAccessToken()

  def __generateJWT(self,pem_file):

    # iss is the app id
    # Ensuring that we request an access token that expires after a minute
    payload = { 
        'iat': datetime.datetime.utcnow(),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=60),
        'iss': self.__app_id 
        }
  
    PEM = ""
    if pem_file == "":
      PEM = os.environ.get('PARTHENON_METRICS_APP_PEM')
    else:
      certs = pem.parse_file(pem_file)
      PEM = str(certs[0])

    if PEM == "":
      error_msg = "No permissions enabled for parthenon metrics app, either a pem file needs to "
      "be provided or the PATHENON_METRICS_APP_PEM variable needs to be defined"
      raise Exception(error_msg)
    self.__jwt_token = jwt.encode(payload,PEM, algorithm='RS256').decode("utf-8")

  def __generateInstallationId(self):
    buffer_temp = BytesIO()
    header = [
            'Authorization: Bearer '+str(self.__jwt_token),
            'Accept: ' + self.__api_version
            ]

    c = pycurl.Curl()
    c.setopt(c.URL, 'https://api.github.com/app/installations')
    c.setopt(c.WRITEDATA, buffer_temp)
    c.setopt(c.HTTPHEADER, header)
    c.perform()
    c.close()

    js_obj = json.loads(buffer_temp.getvalue())

    if isinstance(js_obj, list):
        js_obj = js_obj[0]

    # The installation id will be listed at the end of the url path
    self.__install_id = js_obj['html_url'].rsplit('/', 1)[-1]

  def __generateAccessToken(self):
    buffer_temp = BytesIO()
    header = [
            'Authorization: Bearer '+str(self.__jwt_token),
            'Accept: ' + self.__api_version,
            ]

    https_url_access_tokens = "https://api.github.com/app/installations/" + self.__install_id + "/access_tokens"

    c = pycurl.Curl()
    c.setopt(c.HTTPHEADER, header)
    c.setopt(c.URL, https_url_access_tokens)
    c.setopt(c.POST, 1)
    c.setopt(c.VERBOSE, True)
    c.setopt(c.POSTFIELDS, '')
    c.setopt(c.WRITEDATA, buffer_temp)
    c.perform()
    c.close()

    js_obj = json.loads(buffer_temp.getvalue())

    if isinstance(js_obj, list):
        js_obj = js_obj[0]

    self.__access_token = js_obj['token']

    self.__header = [
            'Authorization: token '+self.__access_token,
            'Accept: ' + self.__api_version,
            ]

  def __getBranches(self):
    buffer_temp = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, self.__repo_url + "/branches")
    c.setopt(c.WRITEDATA, buffer_temp)
    c.setopt(c.HTTPHEADER, self.__header)
    c.perform()
    c.close()
    js_obj_list = json.loads(buffer_temp.getvalue())

    self.__branches = []
    self.__branch_current_commit_sha = {}
    for js_obj in js_obj_list:
      self.__branches.append(js_obj['name'])
      self.__branch_current_commit_sha.update({js_obj['name'] : js_obj['commit']['sha']})

  def getBranches(self):
    if not self.__branches:
      self.__getBranches() 
    
    return self.__branches

  def branchExist(self,branch):
    branches = getBranches()
    if branch in branches:
      return True
    return False

  def refreshBranchCache(self):
    self.__getBranches()

  def createBranch(self,branch, branch_to_fork_from = None):
    """
    Will create a branch if it does not already exists, if the branch does exist
    will do nothing, 

    The new branch will be created by forking it of the latest commit of the default branch
    """
    if branch_to_fork_from is None:
      branch_to_fork_from = self.__default_branch
    branches = self.getBranches()
    if branch in branches:
      return

    if not branch_to_fork_from in branches:
      error_msg = "Cannot create new branch: " + branch + " from " + branch_to_fork_from + " because " + branch_to_fork_from + " does not exist."
      raise Exception(error_msg)

    buffer_temp = BytesIO()
    custom_data = {"ref": "refs/heads/" + branch, "sha": self.__branch_current_commit_sha[branch_to_fork_from]}
    c = pycurl.Curl()
    c.setopt(c.URL, self.__repo_url + '/git/refs')
    c.setopt(c.POST, 1)
    buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
    c.setopt(c.READDATA, buffer_temp2)
    c.setopt(c.WRITEDATA, buffer_temp)
    c.setopt(c.HTTPHEADER, self.__header)
    c.perform()
    c.close()

  def getContents(self,branch=None):
    if branch is None:
      branch = self.__default_branch
    buffer_temp = BytesIO()
    # 1. Check if file exists if so get SHA
    custom_data = {"branch":branch}
    c = pycurl.Curl()
    c.setopt(c.URL, self.__repo_url + '/contents?ref=' + branch)
    buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
    c.setopt(c.READDATA, buffer_temp2)
    c.setopt(c.WRITEDATA, buffer_temp)
    c.setopt(c.HTTPHEADER, self.__header)
    c.perform()
    c.close()

    js_obj = json.loads(buffer_temp.getvalue())

    contents = {}
    if isinstance(js_obj, list):
        # Cycle through list to try to find the right object
        for obj in js_obj:
          contents[obj['name']] = obj['sha']

    return contents

  def upload(self, file_name, branch = None):
    """
    This method attempts to upload a file to the specefied branch

    If the file is found to already exist it will be updated 
    """
    if isinstance(file_name,list):
      file_name = file_name[0]
    if branch is None:
      branch = self.__default_branch
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
      if branch != self.__default_image_branch and not self.__ignore:
        print("Note all images will be uploaded to a branch named: " + self.__default_image_branch + " in the main repository.")
        print("Unless the ignore flag is used.")
        branch = self.__default_image_branch
        self.__use_wiki = False
    
    if self.__use_wiki:
      if branch != "master":
        print("Files can only be uploaded to the wiki repositories master branch")
        return
      else:
        if path.exists(self.__parthenon_wiki_dir + "/" + os.path.basename(os.path.normpath(file_name))):
          commit_msg = "Updating file " + file_name
        else:
          commit_msg = "Adding file " + file_name
        shutil.copy(file_name,self.__parthenon_wiki_dir)
        repo = self.getWikiRepo(branch)
        repo.index.add([str(self.__parthenon_wiki_dir + "/" + os.path.basename(os.path.normpath(file_name)))])
        repo.index.commit(commit_msg)
        repo.git.push("--set-upstream","origin",repo.head.reference)
        return
    else:
      branches = self.getBranches()

      if self.__create_branch:
        self.createBranch(branch)
      elif not branch in branches:
        error_msg = "branch: " + branch + " does not exist in repository."
        raise Exception(error_msg)
      
      contents = self.getContents(branch)

      file_found = False
      if file_name in contents:
        file_found = True

      # 2. convert file into base64 format
      # b is needed if it is a png or image file/ binary file
      data = open(file_name, "rb").read()
      encoded_file = base64.b64encode(data)

      # 3. upload the file, overwrite if exists already
      if file_found:
          custom_data = {
              'message': self.__name + " overwriting file " + file_name,
              'name': self.__name,
              'branch': branch,
              'sha': contents[file_name],
              'content': encoded_file.decode('ascii')
                  }

      else:
          custom_data = {
              'message': self.__name + " uploading file " + file_name,
              'name': self.__name,
              'content': encoded_file.decode('ascii'),
              'branch': branch
                  }

      https_url_to_file = self.__repo_url + "/contents/" + file_name
      c2 = pycurl.Curl()
      c2.setopt(c2.HTTPHEADER, self.__header)
      c2.setopt(c2.URL, https_url_to_file)
      c2.setopt(c2.UPLOAD, 1)
      c2.setopt(c2.VERBOSE, True)
      buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
      c2.setopt(c2.READDATA, buffer_temp2)
      buffer_temp3 = BytesIO()
      c2.setopt(c2.WRITEDATA, buffer_temp3)
      c2.perform()
      c2.close()

  def __fillTree(self, current_node, branch):
      nodes = current_node.getNodes()
      for node in nodes:
        buffer_temp = BytesIO()
        custom_data = {"branch": branch}
        buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
        c = pycurl.Curl()
        c.setopt(c.URL, self.__repo_url + "/contents/" + node.getPath())
        c.setopt(c.READDATA, buffer_temp2)
        c.setopt(c.WRITEDATA, buffer_temp)
        c.setopt(c.HTTPHEADER, self.__header)
        c.perform()
        c.close()
        js_obj = json.loads(buffer_temp,getvalue())

        if isinstance(js_obj, list):
          for ob in js_obj:
            node.insert(ob['name'],ob['type'])
        else:
            node.insert(js_obj['name'],js_obj['type'])

        self.__fillTree(node, branch)

  def getBranchTree(self, branch, access_token):
      buffer_temp = BytesIO()
      custom_data = {"branch": branch}
      buffer_temp2 = BytesIO(json,dumps(custom_data).encode('utf-8'))
      # 1. Check if file exists
      c = pycurl.Curl()
      c.setopt(c.URL, self.__repo_url + "/contents" )
      c.setopt(c.READDATA, buffer_temp2)
      c.setopt(c.WRITEDATA, buffer_temp)
      c.setopt(c.HTTPHEADER, self.__header)
      c.perform()
      c.close()

      js_obj = json.loads(buffer_temp.getvalue())
      for obj in js_obj:
        self.__parth_root.insert(ob['name'],ob['type'])

      self.__fillTree(self.__parth_root, branch)

  def getWikiRepo(self, branch):
     
    wiki_remote = f"https://{self.__name}:{self.__access_token}@github.com/" + self.__user + "/" + self.__repo_name + ".wiki.git"
    if not os.path.isdir(str(self.__parthenon_wiki_dir)):
      repo = Repo.clone_from(wiki_remote, self.__parthenon_wiki_dir)
    else:
      repo = Repo(self.__parthenon_wiki_dir)
      repo.config_writer().set_value("user","name", self.__name).release()

    os.environ["GIT_PASSWORD"] = self.__access_token
    return repo

def main(**kwargs):

  app = ParthenonApp(
      kwargs.pop('wiki'),
      kwargs.pop('ignore'),
      kwargs.pop('permissions'),
      kwargs.pop('create'))

  branch = kwargs.pop('branch')
  if isinstance(branch,list):
    branch = branch[0]

  if 'upload' in kwargs:
    app.upload(kwargs.pop('upload'), branch)

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
                        default = "develop",
                        help=desc)


    desc = ('Branch to use. Default is develop')
    parser.add_argument('--branch','-b',
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


