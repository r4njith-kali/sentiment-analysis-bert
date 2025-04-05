# src/train.py

import os
import numpy as np
import evaluate 
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments, 
    Trainer,
    set_seed
)
import config
from src.data_loader import load_and_preprocess_data
import torch

print(f"Setting random seed to: {config.SEED}")
set_seed(config.SEED)

print(f"Loading tokenizer: {config.MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME) # Not necessary as it as already loaded in load_and_preprocess_data() func but good practice


print("Loading and preprocessing data...")

tokenized_datasets = load_and_preprocess_data(
    tokenizer_name = config.MODEL_NAME,
    dataset_name = config.DATASET_NAME,
    text_column = config.TEXT_COLUMN,
    label_column = config.LABEL_COLUMN,
    max_length = config.MAX_LENGTH,
    seed = config.SEED
)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of evaluation examples: {len(eval_dataset)}")

# ------ LOAD MODEL --------

num_labels = len(train_dataset.unique("labels"))

print(f"Loading model '{config.MODEL_NAME}' for sequence classification with {num_labels} labels.")

model = AutoModelForSequenceClassification(
    config.MODEL_NAME,
    num_labels=num_labels
)

# Move model to CPU, if GPU avail. more to GPU

print(f"Moving model to device: {config.DEVICE}")
model.to(config.DEVICE)

# DEFINE METRICS

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions, references = labels)

# -----Define Training Arguments-------

# Create a unique output dir for this specific run

run_output_dir = os.path.join(config.OUTPUT_DIR_BASE, f"{config.MODEL_NAME}-finetuned-{config.DATASET_NAME}")
run_logging_dir = os.path.join(config.LOGGING_DIR_BASE, f"{config.MODEL_NAME}-finetuned-{config.DATASET_NAME}")

print(f"Training output will be saved to: {run_output_dir}")
print(f"Logging directory set to: {run_logging_dir}")

training_args = TrainingArguments(
    output_dir = run_output_dir,
    logging_dir = run_logging_dir,
    num_train_epochs = config.config.NUM_EPOCHS,
    per_device_train_batch_size = config.TRAIN_BATCH_SIZE,
    per_device_evail_batch_size = config.EVAL_BATCH_SIZE,
    learning_rate = config.LEARNING_RATE,
    weight_decay = config.WEIGHT_DECAY,
    warmup_steps = config.WARMUP_STEPS,

    evaluation_strategy = "epochs",
    save_strategy = "epochs",
    logging_strategy = "steps",
    logging_steps = 100,
    save_total_limit = 2,
    load_best_model_at_end = True,
    metric_for_best_model = "accuracy",
    greater_is_better = True,

    fp16 = torch.cuda.is_available(),
    seed=config.SEED
)

# Intialize Trainer

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)





