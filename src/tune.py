# src/tune.py

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
from data_loader import load_and_preprocess_data
import torch
import optuna

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

def model_init(trial):
    print(f"Initializing fresh model instance...")

    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels = num_labels
    )

    return model

# DEFINE METRICS

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return metric.compute(predictions=predictions, references = labels)

def optuna_hp_space(trial):
    """Defines the hyperparameter search space for Optuna"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 3),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
    }

# Output Dir

hpo_output_dir = os.path.join(config.OUTPUT_DIR_BASE, "hpo-search")
print(f"Hyperparameter search run outputs will be in sub-dir under: {hpo_output_dir}")

# Base Args (Non Tunable)

base_training_args = TrainingArguments(
    output_dir = hpo_output_dir,
    per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    disable_tqdm=True,
    report_to="none",

    # Other standard args

    logging_strategy = "steps",
    logging_steps = 100,
    save_total_limit=1,

    fp16 = torch.cuda.is_available(),
    seed = config.SEED,
    per_device_train_batch_size = config.TRAIN_BATCH_SIZE
)
# ------ LOAD MODEL --------

num_labels = len(train_dataset.unique("labels"))

print(f"Loading model '{config.MODEL_NAME}' for sequence classification with {num_labels} labels.")

model = AutoModelForSequenceClassification.from_pretrained(
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
    return metric.compute(predictions=predictions, references = labels)

# -----Define Training Arguments-------

# Create a unique output dir for this specific run

run_output_dir = os.path.join(config.OUTPUT_DIR_BASE, f"{config.MODEL_NAME}-finetuned-{config.DATASET_NAME}")
run_logging_dir = os.path.join(config.LOGGING_DIR_BASE, f"{config.MODEL_NAME}-finetuned-{config.DATASET_NAME}")

print(f"Training output will be saved to: {run_output_dir}")
print(f"Logging directory set to: {run_logging_dir}")

training_args = TrainingArguments(
    output_dir = run_output_dir,
    logging_dir = run_logging_dir,
    num_train_epochs = config.NUM_EPOCHS,
    per_device_train_batch_size = config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size = config.EVAL_BATCH_SIZE,
    learning_rate = config.LEARNING_RATE,
    weight_decay = config.WEIGHT_DECAY,
    warmup_steps = config.WARMUP_STEPS,

    evaluation_strategy = "epoch",
    save_strategy = "epoch",
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

print("Starting training...")
trainer.train()

print("Training Finished.")

# Save model to output dir

final_model_path = os.path.join(run_output_dir, "best_model")
print(f"Saving the best model to: {final_model_path}")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print("Best model saved successfully.")

# Evaluate after training 
print("Running final evaluation on the evaluation set...")
eval_results = trainer.evaluate()
print("Final Evaluation Results:")
print(eval_results)