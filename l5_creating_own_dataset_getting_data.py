import os
#  for laptop
os.environ['HF_HOME'] = 'd:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'd:/Large data/qa data/transformers/cache/'

import requests
url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
response = requests.get(url)
print(response)
print(response.status_code)
print(response.json())

from configparser import ConfigParser
cfg=ConfigParser()
cfg.read("config.ini")

GITHUB_TOKEN = cfg['GITHUB']['GITHUB_TOKEN']
print(GITHUB_TOKEN)
headers = {"Authorization":f"token {GITHUB_TOKEN}"}
