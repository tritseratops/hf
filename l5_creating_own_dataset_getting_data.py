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
modified_data_file_name = "datasets-issues.jsonl"
# data_file_name = "datasets-issues-mod.jsonl"
# data_files_path = "D:\\Large data\\json\\drug-reviews-test.jsonl"
# data_files = "dataset_info23143w4231.json"

from configparser import ConfigParser
cfg=ConfigParser()
cfg.read("config.ini")
PROJECT_DIR = cfg['LOCAL']['PROJECT_FOLDER']
print(PROJECT_DIR)



"""
load file via panda
removes columns that cause bugs 'author_association', 'timeline_url', 'reactions', 'performed_via_github_app'
saves as dataset arrow file
"""
def remove_bugged_columns(project_dir, old_datafile_name, new_data_file_name, ):
    # data_file_name = "datasets-issues.jsonl"
    data_files = project_dir + old_datafile_name
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
    dataset = tds.remove_columns([  'author_association', 'timeline_url', 'reactions', 'performed_via_github_app']) #  'comments', 'milestone','active_lock_reason', 'draft', 'pull_request', 'body',"closed_at", "created_at", "updated_at", 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees',  'state_reason'
    print("After column remove:")
    print(dataset)
    # data_file_name = "datasets-issues-wo-closed.jsonl"
    data_files = project_dir+new_data_file_name
    dataset.to_json(f"{data_files}", orient="records", lines=True)
# exit()
original_df_name = "datasets-issues.jsonl"
modified_data_file_name = "datasets-issues-wo-closed.jsonl"
data_files = PROJECT_DIR + modified_data_file_name
remove_bugged_columns(PROJECT_DIR, original_df_name, modified_data_file_name)
issues_dataset = load_dataset("json", data_files=data_files, split="train")
print(issues_dataset)

sample = issues_dataset.shuffle(seed=666).select(range(3))

# print out the URL and pull request entries
# for url, pr in zip(sample["html_url"], sample["pull_request"]):
#     print(f">> URL: {url}")
#     print(f">>Pull request: {pr}\n")

issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)

print(issues_dataset)


