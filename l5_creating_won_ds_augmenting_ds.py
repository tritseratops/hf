import os
#  for laptop
os.environ['HF_HOME'] = 'd:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'd:/Large data/qa data/transformers/cache/'

from configparser import ConfigParser
cfg=ConfigParser()
cfg.read("config.ini")
GITHUB_TOKEN = cfg['GITHUB']['GITHUB_TOKEN']
print(GITHUB_TOKEN)
headers = {"Authorization":f"token {GITHUB_TOKEN}"}
import requests

def get_comments(headers, issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]

print(get_comments(headers, 2792))

PROJECT_DIR = cfg['LOCAL']['PROJECT_FOLDER']
print(PROJECT_DIR)
modified_data_file_name = "datasets-issues-wo-closed.jsonl"
data_files = PROJECT_DIR + modified_data_file_name
from datasets import load_dataset
issues_dataset = load_dataset("json", data_files=data_files, split="train")

issues_with_comments_ds = issues_dataset.map(
    lambda  x: {"comments": get_comments(x[headers, "number"])}
)

print(issues_with_comments_ds)
print(issues_with_comments_ds.head(1))

