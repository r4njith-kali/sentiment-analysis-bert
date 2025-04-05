# src/evaluate.py

import os
import numpy as np
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments, # We still need TrainingArguments for the Trainer setup
    Trainer,
    set_seed
)
from src import config
from src.data_loader import load_and_preprocess_data
import torch

def evaluate_model(model_path, dataset_split="test"):
    """Loads a fine-tuned model and evaluates it on a specified dataset split."""

    print(f"--- Evaluating Model from: {model_path} ---")
    print(f"Using dataset split: {dataset_split}")

    # --- 1. Set Seed ---
    print(f"Setting random seed to: {config.SEED}")
    set_seed(config.SEED)

    # --- 2. Load Tokenizer ---
    print("Loading tokenizer...")
    # Load the tokenizer *associated with the saved model*
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading tokenizer from {model_path}. Trying base model {config.MODEL_NAME}. Error: {e}")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)


    # --- 3. Load and Preprocess Data ---
    # We only need the split we want to evaluate (e.g., 'test')
    print("Loading and preprocessing data...")
    tokenized_datasets = load_and_preprocess_data(
        tokenizer_name=model_path, # Use the potentially fine-tuned tokenizer path
        dataset_name=config.DATASET_NAME,
        text_column=config.TEXT_COLUMN,
        label_column=config.LABEL_COLUMN,
        max_length=config.MAX_LENGTH,
        seed=config.SEED
    )

    if dataset_split not in tokenized_datasets:
        raise ValueError(f"Split '{dataset_split}' not found in the loaded dataset. Available splits: {list(tokenized_datasets.keys())}")

    eval_dataset = tokenized_datasets[dataset_split]
    print(f"Number of evaluation examples: {len(eval_dataset)}")

    # --- 4. Load Fine-tuned Model ---
    num_labels = len(eval_dataset.unique("labels")) # Get num_labels from the eval data
    print(f"Loading fine-tuned model from {model_path} with {num_labels} labels.")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels
    )
    print(f"Moving model to device: {config.DEVICE}")
    model.to(config.DEVICE)

    # --- 5. Define Metrics ---
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # --- 6. Define Dummy Training Arguments (needed for Trainer) ---
    # We only need arguments relevant to evaluation, like batch size and directories (though output dir isn't strictly necessary here)
    # Create a temporary directory for evaluation outputs if needed, or reuse parts of config
    eval_output_dir = os.path.join(config.OUTPUT_DIR_BASE, "eval_temp")
    print(f"Using temporary directory for evaluation artifacts: {eval_output_dir}")

    eval_args = TrainingArguments(
        output_dir=eval_output_dir,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        do_train=False, # Explicitly set do_train to False
        do_eval=True,   # Explicitly set do_eval to True
        fp16=torch.cuda.is_available(),
        seed=config.SEED,
        # report_to="none" # Disable logging integrations like W&B for pure evaluation runs
    )

    # --- 7. Initialize Trainer (for evaluation) ---
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # --- 8. Run Evaluation ---
    print("Starting evaluation...")
    results = trainer.evaluate()

    print(f"\n--- Evaluation Results for {model_path} on {dataset_split} split ---")
    for key, value in results.items():
        print(f"{key}: {value:.4f}") # Format to 4 decimal places

    return results

if __name__ == "__main__":
    # Define the path to the model you want to evaluate
    # This should point to the directory where the best model was saved
    saved_model_path = os.path.join(config.OUTPUT_DIR_BASE, f"{config.MODEL_NAME}-finetuned-{config.DATASET_NAME}", "best_model")

    # Check if the saved model path exists before running
    if os.path.exists(saved_model_path) and os.path.isdir(saved_model_path):
        evaluate_model(model_path=saved_model_path, dataset_split="test")
    else:
        print(f"Error: Saved model directory not found at {saved_model_path}")
        print("Please ensure training was completed successfully and the path is correct.")
        # Optional: Evaluate the base model if fine-tuned not found
        # print("\nEvaluating the base pre-trained model instead...")
        # evaluate_model(model_path=config.MODEL_NAME, dataset_split="test")