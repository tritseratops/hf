# copied from 3_fine-tuning_pretrained_without_accelerator.py
import torch
print("Torch cuda available: ",torch.cuda.is_available())
import tensorflow as tf
print("tensorflow GPU list: ", tf.config.list_physical_devices('GPU'))
# exit()
import os
#  for laptop
os.environ['HF_HOME'] = 'd:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'd:/Large data/qa data/transformers/cache/'


from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "Mentatko/dummy-model"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Prepare for training
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})

# common, without accelerator
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

hf_token = "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"
tokenizer.push_to_hub("dummy-model3", use_auth_token=hf_token)
model.push_to_hub("dummy-model3", use_auth_token=hf_token)
model2_path = "D:\Large data\qa data\dummy model3"
from huggingface_hub import Repository
model_namespace = "Mentatko/dummy-model"
model1_path = "D:\Large data\qa data\dummy model"
repo = Repository(model1_path, clone_from=model_namespace)
repo.git_pull()

model.save_pretrained(model2_path)
tokenizer.save_pretrained(model2_path)
exit()

# To make sure that everything will go smoothly during training, we pass our batch to this model:
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

# The training loop
# ensure we use gpu
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)


# adding progress bar
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()



for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# evaluation of model accuracy
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())

