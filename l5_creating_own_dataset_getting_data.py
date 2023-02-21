import os
#  for laptop
os.environ['HF_HOME'] = 'd:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'd:/Large data/qa data/transformers/cache/'

import requests
# url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
# page = 1
# per_page = 1
# base_url = "https://api.github.com/repos"
# owner="huggingface"
# repo="datasets"
# query = f"issues?page={page}&per_page={per_page}&state=all"
# from configparser import ConfigParser
# cfg=ConfigParser()
# cfg.read("config.ini")
# GITHUB_TOKEN = cfg['GITHUB']['GITHUB_TOKEN']
# print(GITHUB_TOKEN)
# headers = {"Authorization":f"token {GITHUB_TOKEN}"}
# issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
# print(issues)
# print(issues.status_code)
# print(issues.json())

# response = requests.get(url)
# print(response)
# print(response.status_code)
# print(response.json())
# exit()


import time
import math
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm

def fetch_issues(
        owner="huggingface",
        repo="datasets",
        num_issues=10_000,
        rate_limit=5_000,
        issues_path=Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100 # number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages), position=0, desc="i", leave=False, ncols=80): #  colour='green',
        print("Page: ", page)
        # query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}") #, headers=headers
        print(issues)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = [] # clear batch for next iteration
            print(f"Reached Github rate limit. Sleeping for 1 hour...")
            time.sleep(60 * 60 +1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl")

# fetch_issues()
from datasets import load_dataset
data_file_name = "datasets-issues.jsonl"
# data_file_name = "datasets-issues-mod.jsonl"
# data_files_path = "D:\\Large data\\json\\drug-reviews-test.jsonl"
# data_files = "dataset_info23143w4231.json"

from configparser import ConfigParser
cfg=ConfigParser()
cfg.read("config.ini")
PROJECT_DIR = cfg['LOCAL']['PROJECT_FOLDER']
print(PROJECT_DIR)

data_files = PROJECT_DIR+data_file_name
# data_files = {"train": PROJECT_DIR+data_file_name}
print(data_files)


import pandas as pd
df = pd.read_json(data_files, lines=True)
print(df)
columns = df.columns
print(df.columns)
print(df.dtypes)
head = df.head(1)
print(head)
print(head.iloc[0])
for i in range(len(columns)-1):
    print(columns[i], " : ", head[columns[i]])

from datasets import Dataset
tds = Dataset.from_pandas(df)
print(tds)
exit()
issues_dataset = load_dataset("json", data_files=data_files) # , split="train"
print(issues_dataset)