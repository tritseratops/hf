import os

#  for laptop
os.environ['HF_HOME'] = 'd:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'd:/Large data/qa data/transformers/cache/'

ds_file_path = "D:\\Large data\\"
from datasets import load_dataset

def load_and_save_ds_drugs():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/"
    data_files = {"train": "train.csv", "test": "test.csv"}
    drug_dataset  = load_dataset("csv", data_files=url + "drugsCom_raw.zip",  delimiter="\t") #field="data",
    print(drug_dataset)
    drug_dataset.set_format("pandas")
    print(drug_dataset["train"][:3])

    train_df = drug_dataset["train"][:]
    frequences = (
        train_df["condition"]
        .value_counts()
        .to_frame()
        .reset_index()
        .rename(columns={"index": "condition", "condition": "frequency"})
    )

    print(frequences.head())

    from datasets import Dataset

    freq_dataset = Dataset.from_pandas(frequences)
    print(freq_dataset)

    drug_dataset.reset_format()
    print(drug_dataset)

    # creating validation set
    drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
    # rename efault test split into validation
    print(drug_dataset_clean)
    drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
    print(drug_dataset_clean)
    # add the "test" set to our DatasetDict
    # drug_dataset_clean["test"]=drug_dataset["test"] # commented because only train file loaded
    drug_dataset_train_test = drug_dataset_clean["train"].train_test_split(train_size=0.6, seed=42)
    drug_dataset_clean["train"]=drug_dataset_train_test.pop("train")
    drug_dataset_clean["test"]=drug_dataset_train_test.pop("test")
    print(drug_dataset_clean)

    # commented to speed up
    # save_path = ds_file_path+"drug-ds-clean"
    # drug_dataset_clean.save_to_disk(save_path)
    #
    # from datasets import load_from_disk
    # loaded_ds = load_from_disk(save_path)
    # print(loaded_ds)

    save_path = ds_file_path+"json//"
    for split, dataset in drug_dataset_clean.items():
        dataset.to_json(f"{save_path}drug-reviews-{split}.jsonl")

load_and_save_ds_drugs()
ds_file_path = "D:\\Large data\\"
save_path = ds_file_path+"json//"
data_files = {
    "train": save_path+"drug-reviews-train.jsonl",
    "validation": save_path+"drug-reviews-validation.jsonl",
    "test": save_path+"drug-reviews-test.jsonl",
}
print(data_files)
loaded_from_json = load_dataset("json", data_files=data_files)
print(loaded_from_json)