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

    num_labels_for_model = 2 

    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_labels_for_model
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

# Initialize trainer

trainer = Trainer(
    args = base_training_args,
    model_init = model_init,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics = compute_metrics
)

# Run hyperparamter search

print("Starting hyperparameter search with Optuna...")

best_run = trainer.hyperparamter_search(
    hp_space = optuna_hp_space,
    n_trials = 10,
    direction = "maximize",
    backend = "optuna"
)

print("\n--- Hyperparameter Search Complete ---")
print(f"Best Run ID: {best_run.run_id}")

print(f"Best Objective (eval_accuracy): {best_run.objective:.4f}")
print("Best Hyperparameters found:")
for param, value in best_run.hyperparameters.items():
    print(f"- {param}: {value}")


# Train Final Model with Best Hyperparameters found

print("\n--- Training Final Model with Best Hyperparameters ---")
best_params = best_run.hyperparameters

final_output_dir = os.path.join(config.OUTPUT_DIR_BASE, "best-tuned-model")
print(f"Final model output directory: {final_output_dir}")

final_training_args = TrainingArguments(
    output_dir=final_output_dir,
    learning_rate=best_params["learning_rate"],
    num_train_epochs=best_params["num_train_epochs"],
    weight_decay=best_params.get("weight_decay", config.WEIGHT_DECAY), 
    warmup_steps=best_params.get("warmup_steps", config.WARMUP_STEPS),

    per_device_train_batch_size=best_params.get("per_device_train_batch_size", config.TRAIN_BATCH_SIZE),

    # --- Copy other necessary fixed arguments ---
    per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_strategy="steps",
    logging_steps=100, 
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    seed=config.SEED,
    report_to="all", 
    disable_tqdm=False, 
)


final_trainer = Trainer(
    model_init=model_init, # Still use model_init for a fresh start
    args=final_training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting final training run...")
final_trainer.train()

print(f"Saving final best model to: {final_output_dir}")
final_trainer.save_model(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

print("Hyperparameter tuning and final model training complete.")

