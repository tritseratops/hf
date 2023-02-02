import os

#  for laptop
os.environ['HF_HOME'] = 'd:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'd:/Large data/qa data/transformers/cache/'

ds_file_path = "D:\\Large data\\"
from datasets import load_dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/"
drug_dataset  = load_dataset("csv", data_files=url + "drugsCom_raw.zip",  delimiter="\t") #field="data",
# data_files = {
#     "train": url + "SQuAD_it-train.json.gz",
#     "test": url + "SQuAD_it-test.json.gz",
# }
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(drug_dataset)

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
print(drug_sample[:3])
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

# rename column 0 to patient ID, as it shoudl be
drug_dataset = drug_dataset.rename_column(original_column_name="Unnamed: 0", new_column_name="patient_id")
print(drug_dataset)

def lowercase_condition(example):
    return {"condition":example["condition"].lower()}

def filter_nones(x):
    return x["condition"] is not None

drug_dataset  = drug_dataset.filter(filter_nones)
# drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
drug_dataset = drug_dataset.map(lowercase_condition)
print(drug_dataset["train"]["condition"][:3])

def computer_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset = drug_dataset.map(computer_review_length)

# Inspect the first training example
print(drug_dataset["train"][0])

print(drug_dataset["train"].sort("review_length")[:3])


def filter_small_reviews(example):
    return example["review_length"]>30

print(drug_dataset.num_rows)
drug_dataset = drug_dataset.filter(filter_small_reviews)
# drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

import html
# drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})
# batched
new_drug_dataset = drug_dataset.map(lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True)
