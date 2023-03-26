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

print("overflow_to_sample_mapping:")
print(inputs["overflow_to_sample_mapping"])

inputs = tokenizer(
    raw_dataset["train"][2:6]["question"],
    raw_dataset["train"][2:6]["context"],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

print(f"The 4 examples gave {len(inputs['input_ids'])} features")
print(f"Here is where examples come from {inputs['overflow_to_sample_mapping']} features")

answers = raw_dataset["train"][2:6]["answers"]
start_positions = []
end_positions = []

for i, offset in enumerate(inputs["offset_mapping"]):
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    answer = answers[sample_idx]
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0]+ len(answer["text"][0])
    sequence_ids = inputs.sequence_ids(i)

    # Find the start and end of the context
    idx = 0
    while sequence_ids[idx] != 1:
        idx +=1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # if the answer is not fully inside the context, label is (0,0)
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions.append(0)
        end_positions.append(0)
    else:
        # otherwise its the start and end token positions
        idx = context_start
        while idx<=context_end and offset[idx][0] <= start_char:
            idx += 1
        start_positions.append(idx-1)
        while idx >= context_start and offset[idx][0] <= start_char:
            idx -= 1
        end_positions.append(idx+1)
print(start_positions, end_positions)

# lets compare the theoretical answer with the decoded span of tokens
idx = 0
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

start  = start_positions[idx]
end = end_positions[idx]
labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start : end + 1])

print(f"Theoretical answer: {answer}, labels give: {labeled_answer}")

# lets check answer that is not in context
idx = 4
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

decoded_example = tokenizer.decode(inputs["input_ids"][idx])
print(f"Theoretical answer: {answer}, decoded example: {decoded_example}")

# grouping preporcessing to a function to use in map method
max_length = 384
stride = 128

def preprocess_training_example(examples):
    questions  = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation = "only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end oif the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # if the answer is not fully inside the context label is (0,0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_end and offset[idx][0] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

train_dataset = raw_dataset["train"].map(
    preprocess_training_example,
    batched=True,
    remove_columns=raw_dataset["train"].column_names,
)

print(len(raw_dataset["train"]), len(train_dataset))

# Processing validation data

# set offsets corresponding to the question to none
def preprocess_validation_example(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
         ]

    inputs["example_id"] = example_ids
    return inputs

validation_dataset = raw_dataset["validation"].map(
    preprocess_validation_example,
    batched=True,
    remove_columns=raw_dataset["validation"].column_names,
)

print(len(raw_dataset["validation"]), len(validation_dataset))

# FINE TUNING MODEL WITH KERAS

# post processing
small_eval_set = raw_dataset["validation"].select(range(100))
trained_checkpoint = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation_example,
    batched=True,
    remove_columns=raw_dataset["validation"].column_names,
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# build a batch of a samll validation set and pass it through a model

import tensorflow as tf
# tf.config.experimental.set_memory_growth(gpu, True)
from transformers import TFAutoModelForQuestionAnswering
eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("numpy")

batch = {k: eval_set_for_model[k] for k in eval_set_for_model.column_names}
trained_model = TFAutoModelForQuestionAnswering.from_pretrained(trained_checkpoint)
print(batch)
outputs = trained_model(**batch)

start_logits = outputs.start_logits.numpy()
end_logits = outputs.end_logits.numpy()

# map each example in small_eval_set to the corresponding features in eval_set
import collections

example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(eval_set):
    example_to_features[feature["example_id"]].append(idx)

# selecting best answers
