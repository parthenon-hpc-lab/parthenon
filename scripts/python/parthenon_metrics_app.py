import os
import jwt
import pem
import datetime
import pycurl
import subprocess
import json
import base64
from urllib.parse import urlencode
from io import BytesIO
from shutil import copyfile

from git import Repo
from git import Head


# Place binary files in orphan branches
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

  def __generateJWT(self,pem_file):
    certs = pem.parse_file(pem_file)

    # 88204 is the app id
    # Ensuring that we request an access token that expires after a minute
    payload = { 
        'iat': datetime.datetime.utcnow(),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=60),
        'iss': 88204
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

  def __getBranches(self):
    header = [
            'Authorization: token '+self.__access_token,
            'Accept: ' + self.__api_version,
            ]
    buffer_temp = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, self.__repo_url + "/branches")
    c.setopt(c.WRITEDATA, buffer_temp)
    c.setopt(c.HTTPHEADER. header)
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

    header = [
            'Authorization: token '+self.__access_token,
            'Accept: ' + self.__api_version,
            ]

    buffer_temp = BytesIO()
    custom_data = {"ref": "refs/heads/" + branch, "sha": self.__branch_current_commit_sha[branch_to_fork_from]}
    c = pycurl.Curl()
    c.setopt(c.URL, self.__repo_url + '/git/refs')
    c.setopt(c.POST, 1)
    buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
    c2.setopt(c.READDATA, buffer_temp2)
    c.setopt(c.WRITEDATA, buffer_temp)
    c.setopt(c.HTTPHEADER, header)
    c.perform()
    c.close()

  def getContents(self,branch=self.__default_branch):
    header = [
            'Authorization: token '+self.__access_token,
            'Accept: ' + self.__api_version,
            ]

    buffer_temp = BytesIO()
    # 1. Check if file exists if so get SHA
    custom_data = {"branch":branch}
    c = pycurl.Curl()
    c.setopt(c.URL, self.__repo_url + '/contents' + '?ref=' + branch)
    buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
    c.setopt(c.READDATA, buffer_temp2)
    c.setopt(c.WRITEDATA, buffer_temp)
    c.setopt(c.HTTPHEADER, header)
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



  def uploadFile(self,file_name, branch = self.__default_branch):

    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
      if branch != self.__default_image_branch:
        print("Note all images will be uploaded to an orphan branch named: " + self.__default_image_branch)
        branch = self.__default_image_branch

    branches = getBranches()

    if not branch in branches:
      raise Execption("branch does not exist on repository")
    
    header = [
            'Authorization: token '+self.__access_token,
            'Accept: ' + self.__api_version,
            ]

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
    c2.setopt(c2.HTTPHEADER, header)
    c2.setopt(c2.URL, https_url_to_file)
    c2.setopt(c2.UPLOAD, 1)
    c2.setopt(c2.VERBOSE, True)
    buffer_temp2 = BytesIO(json.dumps(custom_data).encode('utf-8'))
    c2.setopt(c.READDATA, buffer_temp2)
    buffer_temp3 = BytesIO()
    c2.setopt(c2.WRITEDATA, buffer_temp3)
    c2.perform()
    c2.close()

def get_wiki(token):
    full_local_path = "/path/to/repo/"
    username = "test_app"
    password = token
    remote = f"https://{username}:{password}@github.com/JoshuaSBrown/ConwaysGameOfLife.wiki.git"
    if not os.path.isdir("ConwaysGameOfLife.wiki"):
        print("No repo was found")
        repo = Repo.clone_from(remote, "ConwaysGameOfLife.wiki")
    else:
        print("Dir was found")
        repo = Repo("ConwaysGameOfLife.wiki")
        repo.config_writer().set_value("user", "name", username).release()

    os.environ["GIT_PASSWORD"] = password
    return repo
    #process = subprocess.Popen(["git", "clone https://github.com/JoshuaSBrown/ConwaysGameOfLife.wiki.git"], stdout=subprocess.PIPE)    

def add_file_to_wiki(repo, branch, name_of_file):
    """ If the file is an image file it will be added to an orphan branch with the branch name equivalent to the file name"""
    if name_of_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        branch_name = name_of_file.replace("\.","_")
        git = repo.git
        hexshas = git.log('--pretty=%H','--follow','--',name_of_file).split('\n') 
        print("***********")
        print("Hex sha")
        print(hexshas)
        print("************************")
        repo.git.revert(hexshas, no_edit=True)
        copyfile(name_of_file, "ConwaysGameOfLife.wiki/"+ str(name_of_file))
        #for h in hexshas:
        #    print("Sha is %s" % h)
        repo.index.add([str(name_of_file)])
        repo.index.commit("Test commit message adding image file 2", parent_commits=None)
#        print("Adding orphan branch")
    else:
        repo.index.add([str(name_of_file)])
        repo.index.commit("Adding sample file")

    repo.git.push("--set-upstream","origin", repo.head.reference)


#    os.chdir("ConwaysGameOfLife.wiki/")
#    command = "checkout -b " + str(branch)
#    process = subprocess.Popen(["git",command], stdout=subprocess.PIPE)    
#    command = "git add " + str(name_of_file)
#    process = subprocess.Popen(["git",command], stdout=subprocess.PIPE)    
#    command = 'git commit -m "Adding file"'
#    process = subprocess.Popen(["git",command], stdout=subprocess.PIPE)    
#    command = "git push origin " + str(branch)
#    process = subprocess.Popen(["git",command], stdout=subprocess.PIPE)    
#
     

upload_to_wiki = True
branch = "test"

pem_file = "./jbrown-test-app.2020-12-01.private-key.pem"

jwt_token = generateJSONWebToken(pem_file) #jwt.encode(payload,str(certs[0]), algorithm='RS256')

install_id = getInstallationId(jwt_token)

access_token = createAccessToken(jwt_token, install_id)

name_of_file = "purple_circ.png"

repo = get_wiki(access_token)

print("Trying to print tags")
for tags in repo.tags:
    print("Tag is")
    print(tag)

del os.environ['GIT_PASSWORD']
add_file_to_wiki(repo, branch,name_of_file)
#uploadFile(name_of_file, access_token)

# To add a file to an orphan branch
#        git = repo.git
#        repo.head.reference = Head(repo, 'refs/heads/'+name_of_file)
#        index = repo.index
#        git.reset('--hard')
#        index.add([str(name_of_file)])
#        index.commit("Test commit message adding image file", parent_commits=None)
 
# Execute main function
if __name__ == '__main__':

    parser = argparse.ArgumentParser("python3 parthenon_metrics_app.py -p file.pem")
    
    desc = ('Path to the (permissions file/permissions string) which authenticates the application.')
   
    parser.add_argument('--permissions','-p',
                        type=str,
                        nargs=1,
                        required=True,
                        help=desc)

    args = parser.parse_args()
    try:
        main(**vars(args))
    except Exception:
        raise

def main(**kwargs):

  ParthenonApp app(kwargs.pop('permissions'))

