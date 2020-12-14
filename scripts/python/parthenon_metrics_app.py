import os
import jwt
import pem
import datetime
import pycurl
import json
import base64
from io import BytesIO

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


 Place binary files in orphan branches
# Place text files in wiki main branch
class ParthenonApp:
  def __init__(self, pem_file):
    self.__generateJWT(pem_file)
    self.__generateInstallationId() 
    self.__generateAccessToken()
    self.__name = "Parthenon Github Metrics Application"
    self.__repo_url = "https://api.github.com/repos/lanl/Parthenon"
    self.__default_branch = "develop"
    self.__default_image_branch = "figures"
    self.__branches = []
    self.__branch_current_commit_sha = {}
    self.__api_version = "application/vnd.github.v3+json"
    self.__parth_root = Node()

  def __generateJWT(self,pem_file):
    certs = pem.parse_file(pem_file)

    # iss is the app id
    # Ensuring that we request an access token that expires after a minute
    payload = { 
        'iat': datetime.datetime.utcnow(),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=60),
        'iss': 92734
        }
    
    self.__jwt_token = jwt.encode(payload,str(certs[0]), algorithm='RS256').decode("utf-8")


  def __generateInstallationId(self):
    buffer_temp = BytesIO()
    header = [
            'Authorization: Bearer '+str(self.__jwt_token),
            'Accept: ' + self.__api_version,
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
    c.setopt(c.HTTPHEADER. self.__header)
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

  def createBranch(self,branch, branch_to_fork_from = self.__default_branch):
    branches = self.getBranches()
    if branch in branches:
      raise Exception("Branch already exists cannot create.")

    if not branch_to_fork_from in branches:
      raise Exception("Cannot create new branch: " + branch + " from " + branch_to_fork_from " because " + branch_to_fork_from + " does not exist.")

    buffer_temp = BytesIO()
    custom_data = {"ref": "refs/heads/" + branch, "sha": self.__branch_current_commit_sha[branch_to_fork_from]}
    c = pycurl.Curl()
    c.setopt(c.URL, self.__repo_url + '/git/refs')
    c.setopt(c.POST, 1)
    buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
    c2.setopt(c.READDATA, buffer_temp2)
    c.setopt(c.WRITEDATA, buffer_temp)
    c.setopt(c.HTTPHEADER, self.__header)
    c.perform()
    c.close()

  def getContents(self,branch=self.__default_branch):
    buffer_temp = BytesIO()
    # 1. Check if file exists if so get SHA
    custom_data = {"branch":branch}
    c = pycurl.Curl()
    c.setopt(c.URL, self.__repo_url + '/contents' + '?ref=' + branch)
    buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
    c.setopt(c.READDATA, buffer_temp2)
    c.setopt(c.WRITEDATA, buffer_temp)
    c.setopt(c.HTTPHEADER, self.__header)
    c.perform()
    c.close()

    js_obj = json.loads(buffer_temp.getvalue())

    contents = []
    if isinstance(js_obj, list):
        # Cycle through list to try to find the right object
        for obj in js_obj:
          contents.append(obj['name'])

    return contents

    file_found = False

    if isinstance(js_obj, list):
        # Cycle through list to try to find the right object
        for obj in js_obj:
            if obj['name'] == file_name:
                file_found = True
                break



  def upload(self,file_name, branch = self.__default_branch):

    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
      if branch != self.__default_image_branch:
        print("Note all images will be uploaded to a branch named: " + self.__default_image_branch)
        branch = self.__default_image_branch

    branches = getBranches()

    if not branch in branches:
      raise Execption("branch does not exist on repository")
    
    contents = getContents(branch)

    file_found = False
    if file_name in contents:
      file_found = True
      break

    # 2. convert file into base64 format
    # b is needed if it is a png or image file/ binary file
    data = open(name_of_file, "rb").read()
    encoded_file = base64.b64encode(data)

    # 3. upload the file, overwrite if exists already
    if file_found:
        custom_data = {
            'message': self.__name + " overwriting file " + name_of_file,
            'name': self.__name,
            'branch': branch
            'sha': obj['sha'],
            'content': encoded_file.decode('ascii')
                }

    else:
        custom_data = {
            'message': self.__name + " uploading file " + name_of_file
            'name': self.__name,
            'content': encoded_file.decode('ascii')
                }

    https_url_to_file = self.__repo_url + "/contents/" + name_of_file
    c2 = pycurl.Curl()
    c2.setopt(c2.HTTPHEADER, self.__header)
    c2.setopt(c2.URL, https_url_to_file)
    c2.setopt(c2.UPLOAD, 1)
    c2.setopt(c2.VERBOSE, True)
    buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
    c2.setopt(c.READDATA, buffer_temp2)
    buffer_temp3 = BytesIO()
    c2.setopt(c2.WRITEDATA, buffer_temp3)
    c2.perform()
    c2.close()

def __fillTree(current_node, branch):
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

def getBranchTree(branch, access_token):
    buffer_temp = BytesIO()
    custom_data = {"branch": branch)
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

def main(**kwargs):

  ParthenonApp app(kwargs.pop('permissions'))

  branch = kwargs.pop('branch')
  if 'update' in kwargs:
    app.upload(kwargs.pop('update'), branch)

# Execute main function
if __name__ == '__main__':

    parser = argparse.ArgumentParser("python3 parthenon_metrics_app.py -p file.pem")
    
    desc = ('Path to the (permissions file/permissions string) which authenticates the application.')
   
    parser.add_argument('--permissions','-p',
                        type=str,
                        nargs=1,
                        required=True,
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

    args = parser.parse_args()
    try:
        main(**vars(args))
    except Exception:
        raise


