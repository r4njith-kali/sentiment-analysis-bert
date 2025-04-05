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

# Define Training Arguments

