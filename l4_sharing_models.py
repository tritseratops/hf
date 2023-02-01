import os

#  for laptop
os.environ['HF_HOME'] = 'd:/Large data/qa data/hf_home/'
os.environ['TRANSFORMERS_CACHE'] = 'd:/Large data/qa data/transformers/cache/'

# from transformers import pipeline
#
# camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
# results = camembert_fill_mask("Le camembert est <mask> :)")
#
# print(results)

# direct model instantiation instead of autoc*alsses
# from transformers import CamembertTokenizer, CamembertForMaskedLM
#
# tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
# model = CamembertForMaskedLM.from_pretrained("camembert-base")

# preferrable from Auto*classes
# from transformers import AutoTokenizer, AutoModelForMaskedLM
#
# checkpoint = "camembert-base"
#
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForMaskedLM.from_pretrained(checkpoint)

hf_token = "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"
# model.push_to_hub("dummy-model", use_auth_token=hf_token)
# tokenizer.push_to_hub("dummy-model", use_auth_token=hf_token)
# from transformers import TrainingArguments
# # push_to_hub=True - pushed model to hub after training
# training_args = TrainingArguments(
#     "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True
# )

# tokenizer.push_to_hub("dummy-model", use_auth_token=hf_token)

# from huggingface_hub import create_repo
# create_repo("dummy-model2", private=True, token=hf_token, repo_type="space", space_sdk="gradio")

# from huggingface_hub import upload_file
#
# upload_file(
#     "<path_to_file>/config.json",
#     path_in_repo="config.json",
#     repo_id="<namespace>/dummy-model",
#     token=hf_token,
# )

# 2023.01.22
# from transformers import AutoModelForMaskedLM, AutoTokenizer
#
# checkpoint = "camembert-base"
#
# model = AutoModelForMaskedLM.from_pretrained(checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# # Do whatever with the model, train it, fine-tune it...
# model_path = "D:\Large data\qa data\dummy model"
# model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)

#2023.01.23
model_path = "D:\Large data\qa data\dummy model"
from huggingface_hub import Repository
model_namespace = "Mentatko/dummy-model"
model_path = "D:\Large data\qa data\dummy model2"
repo = Repository(model_path, clone_from=model_namespace)
repo.git_pull()
from transformers import AutoModelForMaskedLM, AutoTokenizer
checkpoint = "camembert-base"
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
repo.git_add()
repo.git_commit("Add model and tokenizer files")
repo.git_push()
# repo.git_tag()