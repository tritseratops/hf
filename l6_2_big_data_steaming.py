import os
#  for laptop
os.environ['HF_HOME'] = 'd:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'd:/Large data/qa data/transformers/cache/'

from datasets import load_dataset
data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)

print(next(iter(pubmed_dataset_streamed)))

shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10000, seed=42)
print(next(iter(shuffled_dataset)))

dataset_head = pubmed_dataset_streamed.take(5)
print(list(dataset_head))

# skip first 1000
train_dataset = shuffled_dataset.skip(1000)
# take 1000 examples after taht
validation_set = shuffled_dataset.take(1000)

law_dataset_streamed = load_dataset(
    "json",
    data_files="https://the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst",
    split="train",
    streaming=True
)
print(next(iter(law_dataset_streamed)))

from itertools import islice
from datasets import interleave_datasets

combined_dataset = interleave_datasets([pubmed_dataset_streamed.remove_columns("meta"), law_dataset_streamed.remove_columns("meta")])
print(list(islice(combined_dataset,2)))

#piling all togather
base_url = "https://the-eye.eu/public/AI/pile/"
data_files = {
    "train": [base_url  + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
    "validation": base_url + "val.jsonl.zst",
    "test": base_url + "test.jsonl.zst",
}
pile_dataset = load_dataset("json", data_files=data_files, streaming=True)
print(next(iter(pile_dataset["train"])))

