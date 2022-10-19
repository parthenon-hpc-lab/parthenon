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
import logging
import datetime
import filecmp
import pathlib
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
    def __init__(self, dir_name="", rel_path=""):
        """
        Creating a Node object

        dir_name is the name of the directory the node contains information about
        rel_path is the actual path to the directory
        """
        self.dir = dir_name
        self.dirs = []
        self.files = []
        self.misc = []
        self.rel_path = rel_path + dir_name

    def insert(self, content, content_type):
        """
        Record the contents of a directory by inserting it

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
        """Returns a list of all nodes in the current node, which are essentially directories."""
        return self.dirs

    def getPath(self):
        """Get the relative path of the current node."""
        return self.rel_path

    def printTree(self):
        """Print contents of node and all child nodes."""
        self._log.info("Contents in folder: " + self.rel_path)
        for fil in self.files:
            self._log.info("File " + fil)
        for mis in self.misc:
            self._log.info("Misc " + mis)
        for node in self.dirs:
            node.printTree()


class GitHubApp:

    """
    GitHubApp Class

    This class is responsible for authenticating against the parthenon repository and interacting
    with the github api.
    """

    def __init__(self, app_id, name, user, repo_name, path_to_app_instance):
        """
        The app is generic and provides a template, to create an app for a specefic repository the
        following arguments are needed:
        * the app id as provided when it is created on github
        * the name of the app
        * the owner of the repository it controls
        * the name of the repository it controls
        * the location of the github child class, should exist within a repo
        """
        self._app_id = app_id
        self._name = name
        self._user = user
        self._repo_name = repo_name

        self._log = logging.getLogger(self._repo_name)
        self._log.setLevel(logging.INFO)

        fh = logging.FileHandler(self._repo_name + ".log", mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        self._log.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        self._log.addHandler(ch)

        self._config_file_dir = pathlib.Path(__file__).parent.absolute()
        self._config_file_name = "githubapp_" + str(self._app_id) + ".config"
        self._config_file_path = pathlib.Path.joinpath(
            self._config_file_dir, self._config_file_name
        )
        self._log.info("github app located at = {}".format(__file__))
        self._log.info("config file path = {}".format(self._config_file_path))

        # Create an empty config file if one does not exist
        if not pathlib.Path.is_file(self._config_file_path):
            self._log.info("No config file available. Creating one.")
            open(self._config_file_path, "a").close()

    def initialize(
        self,
        use_wiki=False,
        ignore=False,
        pem_file="",
        create_branch=False,
        path_to_repo=None,
    ):
        """
        Sets basic properties of the app should be called before any other methods

        use_wiki - determines if by default commands will refer to the wiki repository
        create_branch - determines if you are giving the application the ability to create new
        branches
        pem_file - this is the authentication file needed to do anything with the github api.
        ignore - if this is set to true than images will not be uploaded to a seperate figures
        branch on the main repository. By default binary files are uploaded to a orphan branch so
        as to prevent bloatting the commit history.

        The initialization method is also responsible for authenticating with github and creating
        an access token. The access token is needed to do any further communication or run any other
        operations on github.
        """
        self._ignore = ignore
        self._use_wiki = use_wiki
        self._repo_url = (
            "https://api.github.com/repos/" + self._user + "/" + self._repo_name
        )
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

        if path_to_repo is not None:
            self._log.info("Checking path to repo: {}".format(path_to_repo))
            # Check that the repo specified is valid
            if os.path.isdir(path_to_repo):
                # Check if we are overwriting an existing repo stored in the config file
                with open(self._config_file_path, "r") as file:
                    line = file.readline()
                    # Print a message if they are different
                    if line != path_to_repo:
                        self._log.info(
                            "Changing repo path from {} to {}".format(
                                line, path_to_repo
                            )
                        )

                with open(self._config_file_path, "w") as file:
                    file.write(path_to_repo)

                self._repo_path = path_to_repo
            else:
                error_msg = "The suggested repository path is not valid:\n{}".format(
                    path_to_repo
                )
                self._log.error(error_msg)
                raise RuntimeError(error_msg)
        else:

            if pathlib.Path.is_file(self._config_file_path):

                self._log.info("Reading config file {}".format(self._config_file_path))
                with open(self._config_file_path, "r") as file:
                    line = file.readline()
                    # Throw an error if the path is not valid
                    if not os.path.isdir(line):
                        error_msg = (
                            "The cached path to your repository is not valid {}".format(
                                line
                            )
                        )
                        self._log.error(error_msg)
                    self._repo_path = line
            else:
                # If no config file exists throw an error
                error_msg = str(
                    "No repository path is known to the parthenon_performance_app.\n"
                    "Please call --repository-path or -rp with the path the repository to register it.\n"
                )
                self._log.error(error_msg)
                raise

        self._parthenon_wiki_dir = os.path.normpath(
            self._repo_path + "/../" + self._repo_name + ".wiki"
        )
        self._log.info("Parthenon wiki dir")
        self._log.info(self._parthenon_wiki_dir)
        if isinstance(pem_file, list):
            self._generateJWT(pem_file[0])
        else:
            self._generateJWT(pem_file)
        self._generateInstallationId()
        self._generateAccessToken()

    def _generateJWT(self, pem_file):
        """
        Generates Json web token

        Method will take the permissions (.pem) file provided and populate the json web token
        attribute
        """
        # iss is the app id
        # Ensuring that we request an access token that expires after a minute
        payload = {
            "iat": datetime.datetime.utcnow(),
            "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=60),
            "iss": self._app_id,
        }

        PEM = ""
        if pem_file == "":
            if "GITHUB_APP_PEM" in os.environ:
                pem_file = os.environ.get("GITHUB_APP_PEM")
            else:
                error_msg = "A pem file has not been specified and GITHUB_APP_PEM env varaible is not defined"
                raise Exception(error_msg)

        self._log.info("File loc %s" % pem_file)
        certs = pem.parse_file(pem_file)
        PEM = str(certs[0])

        if PEM == "":
            error_msg = (
                "No permissions enabled for parthenon metrics app, either a pem file needs to "
                "be provided or the GITHUB_APP_PEM variable needs to be defined"
            )
            raise Exception(error_msg)
        self._jwt_token = jwt.encode(payload, PEM, algorithm="RS256")
        # Older instances of jwt return bytestrings, not strings
        if isinstance(self._jwt_token, bytes):
            self._jwt_token = self._jwt_token.decode("utf-8")

    @staticmethod
    def _PYCURL(header, url, option=None, custom_data=None):
        buffer_temp = BytesIO()
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(pycurl.VERBOSE, 0)
        c.setopt(c.WRITEDATA, buffer_temp)
        c.setopt(c.HTTPHEADER, header)
        if option == "POST":
            c.setopt(c.POST, 1)
            c.setopt(c.POSTFIELDS, json.dumps(custom_data))
            c.setopt(c.POSTFIELDSIZE, len(json.dumps(custom_data)))
        elif option == "PUT":
            c.setopt(c.PUT, 1)

        if custom_data is not None:
            buffer_temp2 = BytesIO(json.dumps(custom_data).encode("utf-8"))
            c.setopt(c.READDATA, buffer_temp2)

        c.perform()
        c.close()

        return json.loads(buffer_temp.getvalue())

    def _generateInstallationId(self):
        """
        Generate an installation id

        This method will populate the installation id attribute using the internally stored json
        web token.
        """
        header = [
            "Authorization: Bearer " + str(self._jwt_token),
            "Accept: " + self._api_version,
        ]

        print(header)
        js_obj, _ = self._PYCURL(header, "https://api.github.com/app/installations")
        print(js_obj)

        if isinstance(js_obj, list):
            js_obj = js_obj[0]

        # The installation id will be listed at the end of the url path
        self._install_id = js_obj["html_url"].rsplit("/", 1)[-1]

    def _generateAccessToken(self):
        """
        Creates an access token

        This method will populate the installation attribute using the installation id. The token
        is needed to authenticate any actions run by the application.
        """
        header = [
            "Authorization: Bearer " + str(self._jwt_token),
            "Accept: " + self._api_version,
        ]

        https_url_access_tokens = (
            "https://api.github.com/app/installations/"
            + self._install_id
            + "/access_tokens"
        )

        js_obj = self._PYCURL(header, https_url_access_tokens, option="POST")

        if isinstance(js_obj, list):
            js_obj = js_obj[0]

        self._access_token = js_obj["token"]

        self._header = [
            "Authorization: token " + self._access_token,
            "Accept: " + self._api_version,
        ]

    def _fillTree(self, current_node, branch):
        """
        Creates a content tree of the branch

        This is an internal method that is meant to be used recursively to grab the contents of a
        branch of a remote repository.
        """
        nodes = current_node.getNodes()
        for node in nodes:

            js_obj = self._PYCURL(
                self._header,
                self._repo_url + "/contents/" + node.getPath(),
                custom_data={"branch": branch},
            )

            if isinstance(js_obj, list):
                for ob in js_obj:
                    node.insert(ob["name"], ob["type"])
            else:
                node.insert(js_obj["name"], js_obj["type"])

            self._fillTree(node, branch)

    def _getBranches(self):
        """Internal method for getting a list of the branches that are available on github."""
        page_found = True
        page_index = 1
        self._branches = []
        self._branch_current_commit_sha = {}
        while page_found:
            page_found = False
            js_obj_list = self._PYCURL(
                self._header, self._repo_url + "/branches?page={}".format(page_index)
            )
            page_index = page_index + 1
            for js_obj in js_obj_list:
                page_found = True
                self._branches.append(js_obj["name"])
                self._branch_current_commit_sha.update(
                    {js_obj["name"]: js_obj["commit"]["sha"]}
                )

    def getBranchMergingWith(self, branch):
        """Gets the name of the target branch of `branch` which it will merge with."""
        js_obj_list = self._PYCURL(self._header, self._repo_url + "/pulls")
        self._log.info(
            "Checking if branch is open as a pr and what branch it is targeted to merge with.\n"
        )
        self._log.info("Checking branch %s\n" % (self._user + ":" + branch))
        for js_obj in js_obj_list:
            self._log.info("Found branch: %s.\n" % js_obj.get("head").get("label"))
            if js_obj.get("head").get("label") == self._user + ":" + branch:
                return js_obj.get("base").get("label").split(":", 1)[1]
        return None

    # Public Methods

    def getBranches(self):
        """
        Gets the branches of the repository

        This method will check to see if branches have already been collected from the github
        RESTful api. If the branch tree has not been collected it will update the branches
        attribute.
        """
        if not self._branches:
            self._getBranches()

        return self._branches

    def getLatestCommitSha(self, target_branch):
        """Does what it says gets the latest commit sha for the taget_branch."""
        if not self._branches:
            self._getBranches()
        return self._branch_current_commit_sha.get(target_branch)

    def branchExist(self, branch):
        """
        Determine if branch exists

        This method will determine if a branch exists on the github repository by pinging the
        github api.
        """
        return branch in self.getBranches()

    def refreshBranchCache(self):
        """ "
        Method forces an update of the localy stored branch tree.

        Will update regardless of whether the class already contains a local copy. Might be
        necessary if the remote github repository is updated.
        """
        self._getBranches()

    def createBranch(self, branch, branch_to_fork_from=None):
        """
        Creates a git branch

        Will create a branch if it does not already exists, if the branch does exist
        will do nothing. The new branch will be created by forking it of the latest
        commit of the default branch
        """
        if branch_to_fork_from is None:
            branch_to_fork_from = self._default_branch
        if self.branchExist(branch):
            return

        if not self.branchExist(branch_to_fork_from):
            error_msg = (
                "Cannot create new branch: "
                + branch
                + " from "
                + branch_to_fork_from
                + " because "
                + branch_to_fork_from
                + " does not exist."
            )
            raise Exception(error_msg)

        self._PYCURL(
            self._header,
            self._repo_url + "/git/refs",
            option="POST",
            custom_data={
                "ref": "refs/heads/" + branch,
                "sha": self._branch_current_commit_sha[branch_to_fork_from],
            },
        )

    def getContents(self, branch=None):
        """
        Returns the contents of a branch

        Returns the contents of a branch as a dictionary, where the key is the content and the value
        is the sha of the file/folder etc.
        """
        if branch is None:
            branch = self._default_branch
        # 1. Check if file exists if so get SHA
        js_obj = self._PYCURL(
            self._header,
            self._repo_url + "/contents?ref=" + branch,
            custom_data={"branch": branch},
        )

        contents = {}
        if isinstance(js_obj, list):
            # Cycle through list to try to find the right object
            for obj in js_obj:
                contents[obj["name"]] = obj["sha"]

        return contents

    def upload(self, file_name, branch=None, use_wiki=False, wiki_state="hard"):
        """
        This method attempts to upload a file to the specified branch.

        If the file is found to already exist it will be updated. Image files will by default be placed
        in a figures branch of the main repository, so as to not bloat the repositories commit history.
        """

        # Will only be needed if we are creating a branch
        branch_to_fork_from = self._default_branch

        if isinstance(file_name, list):
            file_name = file_name[0]
        if branch is None:
            branch = self._default_branch
        if file_name.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        ):
            self._log.info("Image file detected")
            if branch != self._default_image_branch and not self._ignore:
                self._log.warning(
                    "Note all images will be uploaded to a branch named: "
                    + self._default_image_branch
                    + " in the main repository."
                )
                self._log.warning("Unless the ignore flag is used.")
                branch = self._default_image_branch
                branch_to_fork_from = "master"
                self._use_wiki = False

        if self._use_wiki or use_wiki:
            if branch != "master":
                error_msg = (
                    "Files can only be uploaded to the wiki repositories master branch"
                )
                raise Exception(error_msg)

            if os.path.exists(
                self._parthenon_wiki_dir
                + "/"
                + os.path.basename(os.path.normpath(file_name))
            ):
                commit_msg = "Updating file " + file_name
            else:
                commit_msg = "Adding file " + file_name
            repo = self.getWikiRepo(branch, wiki_state)
            destination = (
                self._parthenon_wiki_dir
                + "/"
                + os.path.basename(os.path.normpath(file_name))
            )
            if not filecmp.cmp(file_name, destination):
                shutil.copy(file_name, destination)
            repo.index.add(
                [
                    str(
                        self._parthenon_wiki_dir
                        + "/"
                        + os.path.basename(os.path.normpath(file_name))
                    )
                ]
            )
            repo.index.commit(commit_msg)
            repo.git.push("--set-upstream", "origin", repo.head.reference)
            return

        if self._create_branch:
            self.createBranch(branch, branch_to_fork_from)
        elif not self.branchExist(branch):
            error_msg = "branch: " + branch + " does not exist in repository."
            raise Exception(error_msg)

        contents = self.getContents(branch)

        file_found = False
        if os.path.basename(os.path.normpath(file_name)) in contents:
            self._log.warning(
                "File (%s) already exists in branch:%s"
                % (os.path.basename(os.path.normpath(file_name)), branch)
            )
            file_found = True

        # 2. convert file into base64 format
        # b is needed if it is a png or image file/ binary file
        with open(file_name, "rb") as f:
            data = f.read()
        encoded_file = base64.b64encode(data)

        # 3. upload the file, overwrite if exists already
        custom_data = {
            "message": "%s %s file %s"
            % (
                self._name,
                "overwriting" if file_found else "uploading",
                os.path.basename(os.path.normpath(file_name)),
            ),
            "name": self._name,
            "branch": branch,
            "content": encoded_file.decode("ascii"),
        }

        if file_found:
            custom_data["sha"] = contents[os.path.basename(os.path.normpath(file_name))]

        self._log.info(
            "Uploading file (%s) to branch (%s)"
            % (os.path.basename(os.path.normpath(file_name)), branch)
        )
        https_url_to_file = (
            self._repo_url
            + "/contents/"
            + os.path.basename(os.path.normpath(file_name))
        )

        self._PYCURL(self._header, https_url_to_file, "PUT", custom_data)

    def getBranchTree(self, branch):
        """
        Gets the contents of a branch as a tree

        Method will grab the contents of the specified branch from the remote repository. It will
        return the contents as a tree object.
        """
        # 1. Check if file exists
        js_obj = self._PYCURL(
            self._header, self._repo_url + "/contents", "PUT", {"branch": branch}
        )

        for obj in js_obj:
            self._parth_root.insert(obj["name"], obj["type"])

        self._fillTree(self._parth_root, branch)

    def cloneWikiRepo(self, wiki_state="hard"):
        """
        Clone a git repo

        Will clone the wiki repository if it does not exist, if it does exist it will update the
        access permissions by updating the wiki remote url. The repository is then returned.
        """
        wiki_remote = (
            "https://x-access-token:"
            + str(self._access_token)
            + "@github.com/"
            + self._user
            + "/"
            + self._repo_name
            + ".wiki.git"
        )
        if not os.path.isdir(str(self._parthenon_wiki_dir)):
            repo = Repo.clone_from(wiki_remote, self._parthenon_wiki_dir)
        else:
            repo = Repo(self._parthenon_wiki_dir)
            g = git.cmd.Git(self._parthenon_wiki_dir)
            self._log.info("Our remote url is %s" % wiki_remote)
            # git remote show origini
            self._log.info(g.execute(["git", "remote", "show", "origin"]))
            g.execute(["git", "remote", "set-url", "origin", wiki_remote])
            # Ensure local branches are synchronized with server
            g.execute(["git", "fetch"])
            # Will not overwrite files but will reset the index to match with the remote
            if wiki_state == "hard":
                g.execute(["git", "reset", "--hard", "origin/master"])
            elif wiki_state == "mixed":
                g.execute(["git", "reset", "--mixed", "origin/master"])
            elif wiki_state == "soft":
                g.execute(["git", "reset", "--soft", "origin/master"])
            else:
                raise Exception(
                    "Unrecognized github reset option encountered {}".format(wiki_state)
                )

        return repo

    def getWikiRepo(self, branch, wiki_state="hard"):
        """
        Get the git wiki repo

        The github api has only limited supported for interacting with the github wiki, as such the best
        way to do this is to actually clone the github repository and interact with the git repo
        directly. This method will clone the repository if it does not exist. It will then return a
        repo object.
        """
        repo = self.cloneWikiRepo(wiki_state)
        return repo

    def postStatus(
        self, state, commit_sha=None, context="", description="", target_url=""
    ):
        """Post status of current commit."""
        self._log.info("Posting state: %s" % state)
        self._log.info("Posting context: %s" % context)
        self._log.info("Posting description: %s" % description)
        self._log.info("Posting url: %s" % target_url)
        state_list = ["pending", "failed", "error", "success"]
        if state not in state_list:
            raise Exception("Unrecognized state specified " + state)
        if commit_sha is None:
            commit_sha = os.getenv("CI_COMMIT_SHA")
        if commit_sha is None:
            raise Exception(
                "CI_COMMIT_SHA not defined in environment cannot post status"
            )
        custom_data_tmp = {"state": state}
        if context != "":
            custom_data_tmp["context"] = context
        if description != "":
            custom_data_tmp["description"] = description
        if target_url != "":
            custom_data_tmp["target_url"] = target_url

        self._PYCURL(
            self._header,
            self._repo_url + "/statuses/" + commit_sha,
            "POST",
            custom_data_tmp,
        )

    def getStatus(self):
        """Get status of current commit."""
        commit_sha = os.getenv("CI_COMMIT_SHA")
        if commit_sha is None:
            raise Exception(
                "CI_COMMIT_SHA not defined in environment cannot post status"
            )

        # 1. Check if file exists if so get SHA
        js_obj = self._PYCURL(
            self._header, self._repo_url + "/commits/Add_to_dev/statuses"
        )
        return js_obj
