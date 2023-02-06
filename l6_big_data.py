import os
#  for laptop
os.environ['HF_HOME'] = 'd:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'd:/Large data/qa data/transformers/cache/'

from datasets import load_dataset
data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset("json", data_files=data_files, split="train")
print(pubmed_dataset)
print(pubmed_dataset[0])

import psutil
# output memry, convert bytes to mb
print(f"RAM used: {psutil.Process().memory_info().rss: .2f} B")
print(f"RAM used: {psutil.Process().memory_info().rss/ (1024 * 1024): .2f} MB")

print(f"Number of files in dataset: {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size/(1024**3)
print(f"Dataset size (cache file) : {size_gb:.2f} GB")

import timeit
code_snippet = """
batch_size = 1000
for idx in range(0, len(pubmed_dataset), batch_size):
    _=pubmed_dataset[idx:idx + batch_size]
"""

time = timeit.timeit(stmt=code_snippet, number=1, globals=globals())
print(
    f"Iterated over {len(pubmed_dataset)} examples (about {size_gb:.1f} GB) in "
    f"{time:.1f}s, i.e. {size_gb/time:.3f} GB/s"
)

