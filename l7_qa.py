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
small_eval_set = raw_dataset["validation"].select(range(25)) # course number 100
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
# switching to CPU, because GPU has not enough memory
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# decreasing memory fragmentation
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
# sess = tf.Session(config=config)
outputs = trained_model(**batch)

start_logits = outputs.start_logits.numpy()
end_logits = outputs.end_logits.numpy()

# map each example in small_eval_set to the corresponding features in eval_set
import collections

example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(eval_set):
    example_to_features[feature["example_id"]].append(idx)

# selecting best answers
import numpy as np

n_best = 20
max_answer_length = 30
predicted_answers = []

for example in small_eval_set:
    example_id = example["id"]
    context = example["context"]
    answers = []

    for feature_index in example_to_features[example_id]:
        start_logit  = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = eval_set["offset_mapping"][feature_index]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers that are not fully in the context
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # Skip answers that are < 0 or > max_answer_length
                if(
                    end_index - start_index < 0
                    or end_index - start_index > max_answer_length
                ):
                    continue
                answers.append(
                    {
                        "text" : context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score" : start_logit[start_index] + end_logit[end_index],
                    }
                )

        best_answer = max(answers, key=lambda x: x["logit_score"])
        predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})

import evaluate
metric = evaluate.load("squad")

theoretical_answers = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
]

# check if results are correct
print(predicted_answers[0])
print(theoretical_answers[0])

print(metric.compute(predictions=predicted_answers, references=theoretical_answers))

# put all into compute_metrics() function
from tqdm.auto import tqdm

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    predicted_answers = []
    for example in tqdm(examples):
        example_id  =example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for featured_index in example_to_features[example_id]:
            start_logit = start_logits[featured_index]
            end_logit = end_logits[featured_index]
            offsets = features[featured_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length <0 or with the length > max answer length
                    if (
                        end_index < start_index
                        or end_index - start_index +1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text" : context[offsets[start_index][0]  : offsets[end_index][1]],
                        "logit_score" : start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

print(compute_metrics(start_logits, end_logits, eval_set, small_eval_set))

# Fine tuning a model
model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_dataset = model.prepare_tf_dataset(
    train_dataset,
    collate_fn=data_collator,
    shuffle=True,
    batch_size=4,
)
tf_eval_dataset = model.prepare_tf_dataset(
    validation_dataset,
    collate_fn=data_collator,
    shuffle=False,
    batch_size=4
)

# set hyperparameters and compile model
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf

# The number of training steps is a number of smaples in the dataset,
# divided by batch size than multiplied by the total number of epochs
# Note that tf_train_dataset is batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size
num_train_epochs = 1 # 3
num_train_steps = len(tf_train_dataset) * num_train_epochs
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")

from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(output_dir="bert-finetuned-squad", tokenizer=tokenizer)

# we are going to do validation afterwards, so no validation mid-training
model.fit(tf_train_dataset, callbacks=[callback], epochs=num_train_epochs)

# evaluate our model
predictions = model.predict(tf_eval_dataset)
metrics = compute_metrics(
    predictions["start_logits"],
    predictions["end_logits"],
    validation_dataset,
    raw_dataset["validation"],
)