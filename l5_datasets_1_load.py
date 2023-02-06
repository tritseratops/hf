import os

#  for laptop
os.environ['HF_HOME'] = 'd:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'd:/Large data/qa data/transformers/cache/'

ds_file_path = "D:\\Large data\\"
from datasets import load_dataset
# squad_it_dataset = load_dataset("json", data_files=ds_file_path + "SQuAD_it-train.json", field="data")
# print(squad_it_dataset)
# print(squad_it_dataset["train"][0])
# data_files = {"train": ds_file_path+"SQuAD_it-train.json", "test": ds_file_path+"SQuAD_it-test.json"}
# data_files = {"train": ds_file_path+"SQuAD_it-train.json.gz", "test": ds_file_path+"SQuAD_it-test.json.gz"}
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# remote
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset)
