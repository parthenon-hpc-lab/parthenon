#!/usr/bin/env python3
# =========================================================================================
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
# =========================================================================================

import os
import datetime
import filecmp
import pathlib
import pprint
import json
import shutil
import base64
from io import BytesIO
import jwt
import pem
import pycurl
from git import Repo
import git

class Node:

    """
    Class used to store branch contents in terms of files, directories or misc
    """

    def __init__(self, dir_name="", rel_path=""):
        self.dir = dir_name
        self.dirs = []
        self.files = []
        self.misc = []
        self.rel_path = rel_path + dir_name

    def insert(self, content, content_type):
        """
        Will either store new information as a file, directory or misc type.
        If the content type is of type dir than a new node is created.
        """
        if content_type == "dir":
            self.dirs.append(Node(content, self.rel_path + "/"))
        elif content_type == "file":
            self.files.append(content)
        else:
            self.files.append(content)

    def getNodes(self):
        """
        Returns a list of all nodes in the current node, which are essentially
        directories.
        """
        return self.dirs

    def getPath(self):
        """
        Get the relative path of the current node.
        """
        return self.rel_path

    def printTree(self):
        """
        Print contents of node and all child nodes
        """
        print("Contents in folder: " + self.rel_path)
        for fil in self.files:
            print("File " + fil)
        for mis in self.misc:
            print("Misc " + mis)
        for node in self.dirs:
            node.printTree()



class App:

    """
    App Class

    This class is responsible for authenticating against the parthenon repository and interacting
    with the github api.
    """

    def __init__(self, app_id, name, user, repo_name):
        """
        The app is generic and provides a template, to create an app for a specefic repository the
        following arguments are needed:
        * the app id as provided when it is created on github
        * the name of the app
        * the owner of the repository it controls
        * the name of the repository it controls
        """
        self._app_id = app_id
        self._name = name
        self._user = user
        self._repo_name = repo_name

    def initialize(self, use_wiki=False, ignore=False,
                   pem_file="", create_branch=False):
        self._ignore = ignore
        self._use_wiki = use_wiki
        self._repo_url = "https://api.github.com/repos/" + \
            self._user + "/" + self._repo_name
        if isinstance(create_branch, list):
            self._create_branch = create_branch[0]
        else:
            self._create_branch = create_branch
        self._default_branch = "develop"
        self._default_image_branch = "figures"
        self._branches = []
        self._branch_current_commit_sha = {}
        self._api_version = "application/vnd.github.v3+json"
        self._parth_root = Node()
        self._parthenon_home = str(pathlib.Path(__file__).parent.absolute())
        try:
            self._parthenon_home = self._parthenon_home[:self._parthenon_home.rindex(
                self._repo_name) + len("/" + self._repo_name)]
        except Exception:
            error_msg = str(os.path.realpath(
                __file__)) + " must be run from within the " + self._repo_name + " repository."
            print(error_msg)
            raise

        self._parthenon_wiki_dir = os.path.normpath(
            self._parthenon_home + "/../" + self._repo_name + ".wiki")
        print("******** Parthenon wiki dir")
        print(self._parthenon_wiki_dir)
        if isinstance(pem_file, list):
            self._generateJWT(pem_file[0])
        else:
            self._generateJWT(pem_file)
        self._generateInstallationId()
        self._generateAccessToken()

    def _generateJWT(self, pem_file):
        """
        Method will take the permissions (.pem) file provided and populate the json web token attribute
        """
        # iss is the app id
        # Ensuring that we request an access token that expires after a minute
        payload = {
            'iat': datetime.datetime.utcnow(),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=60),
            'iss': self._app_id
        }

        PEM = ""
        if pem_file == "":
            if "GITHUB_APP_PEM" in os.environ:
                pem_file = os.environ.get('GITHUB_APP_PEM')
            else:
                error_msg = "A pem file has not been specified and GITHUB_APP_PEM env varaible is not defined"
                raise Exception(error_msg)

        print("File loc %s" % pem_file)
        certs = pem.parse_file(pem_file)
        PEM = str(certs[0])

        if PEM == "":
            error_msg = ("No permissions enabled for parthenon metrics app, either a pem file needs to "
                         "be provided or the GITHUB_APP_PEM variable needs to be defined")
            raise Exception(error_msg)
        self._jwt_token = jwt.encode(
            payload, PEM, algorithm='RS256').decode("utf-8")

    def _generateInstallationId(self):
        """
        This method will populate the installation id attribute using the internally stored json web token.
        """
        buffer_temp = BytesIO()
        header = [
            'Authorization: Bearer ' + str(self._jwt_token),
            'Accept: ' + self._api_version
        ]

        c = pycurl.Curl()
        c.setopt(c.URL, 'https://api.github.com/app/installations')
        c.setopt(pycurl.VERBOSE, 0)
        c.setopt(pycurl.WRITEFUNCTION, lambda x: None)
        c.setopt(c.WRITEDATA, buffer_temp)
        c.setopt(c.HTTPHEADER, header)
        c.perform()
        c.close()

        js_obj = json.loads(buffer_temp.getvalue())

        if isinstance(js_obj, list):
            js_obj = js_obj[0]

        # The installation id will be listed at the end of the url path
        self._install_id = js_obj['html_url'].rsplit('/', 1)[-1]

    def _generateAccessToken(self):
        """
        This method will populate the installation attribute using the installation id. The token
        is needed to authenticate any actions run by the application.
        """
        buffer_temp = BytesIO()
        header = [
            'Authorization: Bearer ' + str(self._jwt_token),
            'Accept: ' + self._api_version,
        ]

        https_url_access_tokens = "https://api.github.com/app/installations/" + \
            self._install_id + "/access_tokens"

        c = pycurl.Curl()
        c.setopt(pycurl.VERBOSE, 0)
        c.setopt(c.HTTPHEADER, header)
        c.setopt(pycurl.WRITEFUNCTION, lambda x: None)
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

        self._access_token = js_obj['token']

        self._header = [
            'Authorization: token ' + self._access_token,
            'Accept: ' + self._api_version,
        ]

    def _fillTree(self, current_node, branch):
        """
        This is an internal method that is meant to be used recursively to grab the contents of a
        branch of a remote repository.
        """
        nodes = current_node.getNodes()
        for node in nodes:
            buffer_temp = BytesIO()
            custom_data = {"branch": branch}
            buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
            c = pycurl.Curl()
            c.setopt(pycurl.VERBOSE, 0)
            c.setopt(c.URL, self._repo_url + "/contents/" + node.getPath())
            c.setopt(pycurl.WRITEFUNCTION, lambda x: None)
            c.setopt(c.READDATA, buffer_temp2)
            c.setopt(c.WRITEDATA, buffer_temp)
            c.setopt(c.HTTPHEADER, self._header)
            c.perform()
            c.close()
            js_obj = json.loads(buffer_temp.getvalue())

            if isinstance(js_obj, list):
                for ob in js_obj:
                    node.insert(ob['name'], ob['type'])
            else:
                node.insert(js_obj['name'], js_obj['type'])

            self._fillTree(node, branch)

    def _getBranches(self):
        buffer_temp = BytesIO()
        c = pycurl.Curl()
        c.setopt(pycurl.VERBOSE, 0)
        c.setopt(pycurl.WRITEFUNCTION, lambda x: None)
        c.setopt(c.URL, self._repo_url + "/branches")
        c.setopt(c.WRITEDATA, buffer_temp)
        c.setopt(c.HTTPHEADER, self._header)
        c.perform()
        c.close()
        js_obj_list = json.loads(buffer_temp.getvalue())

        self._branches = []
        self._branch_current_commit_sha = {}
        for js_obj in js_obj_list:
            self._branches.append(js_obj['name'])
            self._branch_current_commit_sha.update(
                {js_obj['name']: js_obj['commit']['sha']})

    def getBranchMergingWith(self, branch):
        """Gets the name of the target branch of `branch` which it will merge with."""
        buffer_temp = BytesIO()
        c = pycurl.Curl()
        c.setopt(pycurl.VERBOSE, 0)
        c.setopt(pycurl.WRITEFUNCTION, lambda x: None)
        c.setopt(c.URL, self._repo_url + "/pulls")
        c.setopt(c.WRITEDATA, buffer_temp)
        c.setopt(c.HTTPHEADER, self._header)
        c.perform()
        c.close()
        js_obj_list = json.loads(buffer_temp.getvalue())
        print("Checking if branch is open as a pr and what branch it is targeted to merge with.\n")
        print("Checking branch %s\n" % (self._user + ":" + branch))
        for js_obj in js_obj_list:
            print("Found branch: %s.\n" % js_obj.get('head').get('label'))
            if js_obj.get('head').get('label') == self._user + ":" + branch:
                return js_obj.get('base').get('label').split(':', 1)[1]
        return None

    """
  Public Methods
  """

    def getBranches(self):
        """
        This method will check to see if branches have already been collected from the github RESTful
        api. If the branch tree has not been collected it will update the branches attribute.
        """
        if not self._branches:
            self._getBranches()

        return self._branches

    def getLatestCommitSha(self, target_branch):
        if not self._branches:
            self._getBranches()
        return self._branch_current_commit_sha.get(target_branch)

    def branchExist(self, branch):
        """
        This method will determine if a branch exists on the github repository by pinging the github api
        """
        branches = self.getBranches()
        if branch in branches:
            return True
        return False

    def refreshBranchCache(self):
        """"
        Method forces an update of the localy stored branch tree.
        
        Will update regardless of whether the class already contains a local copy. Might be
        necessary if the remote github repository is updated.
        """
        self._getBranches()

    def createBranch(self, branch, branch_to_fork_from=None):
        """
        Will create a branch if it does not already exists, if the branch does exist
        will do nothing,

        The new branch will be created by forking it of the latest commit of the default branch
        """
        if branch_to_fork_from is None:
            branch_to_fork_from = self._default_branch
        if self.branchExist(branch):
            return

        if not self.branchExist(branch_to_fork_from):
            error_msg = "Cannot create new branch: " + branch + " from " + \
                branch_to_fork_from + " because " + branch_to_fork_from + " does not exist."
            raise Exception(error_msg)

        buffer_temp = BytesIO()
        custom_data = {
            "ref": "refs/heads/" + branch,
            "sha": self._branch_current_commit_sha[branch_to_fork_from]}
        c = pycurl.Curl()
        c.setopt(pycurl.VERBOSE, 0)
        c.setopt(pycurl.WRITEFUNCTION, lambda x: None)
        c.setopt(c.URL, self._repo_url + '/git/refs')
        c.setopt(c.POST, 1)
        buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
        c.setopt(c.READDATA, buffer_temp2)
        c.setopt(c.WRITEDATA, buffer_temp)
        c.setopt(c.HTTPHEADER, self._header)
        c.perform()
        c.close()

    def getContents(self, branch=None):
        """
        Returns the contents of a branch as a dictionary, where the key is the content and the value
        is the sha of the file/folder etc.
        """
        if branch is None:
            branch = self._default_branch
        buffer_temp = BytesIO()
        # 1. Check if file exists if so get SHA
        custom_data = {"branch": branch}
        c = pycurl.Curl()
        c.setopt(pycurl.VERBOSE, 0)
        c.setopt(pycurl.WRITEFUNCTION, lambda x: None)
        c.setopt(c.URL, self._repo_url + '/contents?ref=' + branch)
        buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
        c.setopt(c.READDATA, buffer_temp2)
        c.setopt(c.WRITEDATA, buffer_temp)
        c.setopt(c.HTTPHEADER, self._header)
        c.perform()
        c.close()

        js_obj = json.loads(buffer_temp.getvalue())

        contents = {}
        if isinstance(js_obj, list):
                # Cycle through list to try to find the right object
            for obj in js_obj:
                contents[obj['name']] = obj['sha']

        return contents

    def upload(self, file_name, branch=None, use_wiki=False):
        """
        This method attempts to upload a file to the specefied branch.

        If the file is found to already exist it will be updated. Image files will by default be placed
        in a figures branch of the main repository, so as to not bloat the repositories commit history.
        """
        if isinstance(file_name, list):
            file_name = file_name[0]
        if branch is None:
            branch = self._default_branch
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            print("Image file detected")
            if branch != self._default_image_branch and not self._ignore:
                print(
                    "Note all images will be uploaded to a branch named: " +
                    self._default_image_branch +
                    " in the main repository.")
                print("Unless the ignore flag is used.")
                branch = self._default_image_branch
                self._use_wiki = False

        if self._use_wiki or use_wiki:
            if branch != "master":
                print(
                    "Files can only be uploaded to the wiki repositories master branch")
                return
            else:
                if os.path.exists(self._parthenon_wiki_dir + "/" +
                                  os.path.basename(os.path.normpath(file_name))):
                    commit_msg = "Updating file " + file_name
                else:
                    commit_msg = "Adding file " + file_name
                repo = self.getWikiRepo(branch)
                destination = self._parthenon_wiki_dir + "/" + \
                    os.path.basename(os.path.normpath(file_name))
                if not filecmp.cmp(file_name, destination):
                    shutil.copy(file_name, destination)
                repo.index.add([str(self._parthenon_wiki_dir + "/" +
                                    os.path.basename(os.path.normpath(file_name)))])
                repo.index.commit(commit_msg)
                repo.git.push("--set-upstream", "origin", repo.head.reference)
                return
        else:
            if self._create_branch:
                self.createBranch(branch)
            elif not self.branchExist(branch):
                error_msg = "branch: " + branch + " does not exist in repository."
                raise Exception(error_msg)

            contents = self.getContents(branch)

            file_found = False
            if os.path.basename(os.path.normpath(file_name)) in contents:
                print("File (%s) already exists in branch:%s" %
                      (os.path.basename(os.path.normpath(file_name)), branch))
                file_found = True

            # 2. convert file into base64 format
            # b is needed if it is a png or image file/ binary file
            with open(file_name, "rb") as f:
              data = f.read()
            encoded_file = base64.b64encode(data)

            # 3. upload the file, overwrite if exists already
            if file_found:
                custom_data = {
                    'message': self._name + " overwriting file " + os.path.basename(os.path.normpath(file_name)),
                    'name': self._name,
                    'branch': branch,
                    'sha': contents[os.path.basename(os.path.normpath(file_name))],
                    'content': encoded_file.decode('ascii')
                }

            else:
                custom_data = {
                    'message': self._name + " uploading file " + os.path.basename(os.path.normpath(file_name)),
                    'name': self._name,
                    'content': encoded_file.decode('ascii'),
                    'branch': branch
                }

            print("Uploading file (%s) to branch (%s)" %
                  (os.path.basename(os.path.normpath(file_name)), branch))
            https_url_to_file = self._repo_url + "/contents/" + \
                os.path.basename(os.path.normpath(file_name))
            c2 = pycurl.Curl()
            c2.setopt(pycurl.VERBOSE, 0)
            c2.setopt(pycurl.WRITEFUNCTION, lambda x: None)
            c2.setopt(c2.HTTPHEADER, self._header)
            c2.setopt(c2.URL, https_url_to_file)
            c2.setopt(c2.PUT, 1)
            c2.setopt(c2.VERBOSE, True)
            buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
            c2.setopt(c2.READDATA, buffer_temp2)
            buffer_temp3 = BytesIO()
            c2.setopt(c2.WRITEDATA, buffer_temp3)
            c2.perform()
            c2.close()

    def getBranchTree(self, branch):
        """
        Method will grab the contents of the specified branch from the remote repository. It will
        return the contents as a tree object.
        """
        buffer_temp = BytesIO()
        custom_data = {"branch": branch}
        buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
        # 1. Check if file exists
        c = pycurl.Curl()
        c.setopt(pycurl.VERBOSE, 0)
        c.setopt(pycurl.WRITEFUNCTION, lambda x: None)
        c.setopt(c.URL, self._repo_url + "/contents")
        c.setopt(c.READDATA, buffer_temp2)
        c.setopt(c.WRITEDATA, buffer_temp)
        c.setopt(c.HTTPHEADER, self._header)
        c.perform()
        c.close()

        js_obj = json.loads(buffer_temp.getvalue())
        for obj in js_obj:
            self._parth_root.insert(obj['name'], obj['type'])

        self._fillTree(self._parth_root, branch)

    def cloneWikiRepo(self):
        wiki_remote = "https://x-access-token:" + \
            str(self._access_token) + "@github.com/" + \
            self._user + "/" + self._repo_name + ".wiki.git"
        if not os.path.isdir(str(self._parthenon_wiki_dir)):
            repo = Repo.clone_from(wiki_remote, self._parthenon_wiki_dir)
        else:
            repo = Repo(self._parthenon_wiki_dir)
            g = git.cmd.Git(self._parthenon_wiki_dir)
            print("Our remote url is %s" % wiki_remote)
            print(g.execute(['git','remote','show','origin']))  # git remote show origini
            g.execute(['git','remote','set-url','origin',wiki_remote])
        return repo

    def getWikiRepo(self, branch):
        """
        The github api has only limited supported for interacting with the github wiki, as such the best
        way to do this is to actually clone the github repository and interact with the git repo
        directly. This method will clone the repository if it does not exist. It will then return a
        repo object.
        """
        repo = self.cloneWikiRepo()
        return repo

    def postStatus(self, state, commit_sha=None, context="",
                   description="", target_url=""):
        """
        Post status of current commit.
        """
        state_list = ['pending', 'failed', 'error', 'success']
        if state not in state_list:
            raise Exception("Unrecognized state specified " + state)
        if commit_sha is None:
            commit_sha = os.getenv('CI_COMMIT_SHA')
        if commit_sha is None:
            raise Exception(
                "CI_COMMIT_SHA not defined in environment cannot post status")
        buffer_temp = BytesIO()
        custom_data = {"state": state}
        if context != "":
            custom_data["context"] = context
        if description != "":
            custom_data["description"] = description
        if target_url != "":
            custom_data["target_url"] = target_url
        c = pycurl.Curl()
        c.setopt(c.URL, self._repo_url + '/statuses/' + commit_sha)
        c.setopt(c.POST, 1)
        c.setopt(pycurl.WRITEFUNCTION, lambda x: None)
        buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
        c.setopt(c.READDATA, buffer_temp2)
        c.setopt(c.WRITEDATA, buffer_temp)
        c.setopt(c.HTTPHEADER, self._header)
        c.perform()
        c.close()
        js_obj = json.loads(buffer_temp.getvalue())
        pprint.pprint(json.dumps(js_obj, indent=2))

    def getStatus(self):
        """
        Get status of current commit.
        """
        commit_sha = os.getenv('CI_COMMIT_SHA')
        if commit_sha is None:
            raise Exception(
                "CI_COMMIT_SHA not defined in environment cannot post status")

        buffer_temp = BytesIO()
        # 1. Check if file exists if so get SHA
        c = pycurl.Curl()
        c.setopt(pycurl.VERBOSE, 0)
        c.setopt(pycurl.WRITEFUNCTION, lambda x: None)
        c.setopt(c.URL, self._repo_url + '/commits/Add_to_dev/statuses')
        c.setopt(c.WRITEDATA, buffer_temp)
        c.setopt(c.HTTPHEADER, self._header)
        c.perform()
        c.close()

        js_obj = json.loads(buffer_temp.getvalue())
        return js_obj
