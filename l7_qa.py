import os
#  for laptop
os.environ['HF_HOME'] = 'd:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'd:/Large data/qa data/transformers/cache/'

from datasets import load_dataset
raw_dataset = load_dataset("squad")

print(raw_dataset)

print("Context: ", raw_dataset["train"][0]["context"])
print("Question: ", raw_dataset["train"][0]["question"])
print("Answer: ", raw_dataset["train"][0]["answers"])

print(raw_dataset["train"].filter(lambda x: len(x["answers"]["text"]) != 1))

print(raw_dataset["validation"][0]["answers"])
print(raw_dataset["validation"][2]["answers"])
print(raw_dataset["validation"][2]["context"])
print(raw_dataset["validation"][2]["question"])

from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print(tokenizer.is_fast)

context = raw_dataset["train"][0]["context"]
question = raw_dataset["train"][0]["question"]

inputs = tokenizer(question, context)

print(tokenizer.decode(inputs["input_ids"]))

inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True
)

print(inputs.keys())
for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))