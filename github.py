import os 
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv();
  
owner = "techwithtim"
repo = "Flask-Web-App-Tutorial"
endpoint = 'issues'



github_token = os.getenv("GITHUB_TOKEN")
def fetch_github(owner,repo,endpoint):
  url = f"https://api.github.com/repos/{owner}/{repo}/{endpoint}"
  headers = {
    "Authorization": f"Bearer {github_token}"
  }
  response = requests.get(url,headers=headers)
  if requests.status_codes == 200:
    data = response.json();
  else:
    print("Fail with status code: ", response.status_code)
    return []
  print(data)
  return data


def load_issues(issues):
  docs = []
  for entry in issues:
    metadata = {
      "author": entry["user"]["login"],
      "comments": entry["comments"],
      "body": entry["body"],
      "labels": entry["labels"],
      "created_at": entry["created_at"]
    }
    data = entry["title"]
    if entry["body"]:
      data += entry["body"]

    doc = Document(page_content=data, metadata=metadata)
    docs.append(doc)
  return docs

def fetch_github_issues(owner,repo):
  data = fetch_github(owner,repo,"issues")
  return load_issues(data)




