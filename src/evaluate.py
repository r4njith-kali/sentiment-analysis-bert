# src/evaluate.py

import os
import numpy as np
import evaluate # Hugging Face's library for metrics
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments, # We still need TrainingArguments for the Trainer setup
    Trainer,
    set_seed
)
from src import config # Your configuration file
from src.data_loader import load_and_preprocess_data # Your data loading function
import torch # Import torch to check device

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
    # Use trust_remote_code=True if model used custom code, unlikely for bert-base
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Warning: Error loading tokenizer from {model_path}. Trying base model {config.MODEL_NAME}. Error: {e}")
        print("This might happen if tokenizer files weren't saved correctly with the model.")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)


    # --- 3. Load and Preprocess Data ---
    # We only need the split we want to evaluate (e.g., 'test')
    print("Loading and preprocessing data...")
    # Use the *loaded* tokenizer from the model path for consistency
    tokenized_datasets = load_and_preprocess_data(
        tokenizer_name=model_path, # Use the potentially fine-tuned tokenizer path itself
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
    # Use trust_remote_code=True if model used custom code, unlikely for bert-base
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels
    )
    effective_device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Moving model to device: {effective_device}")
    model.to(effective_device)

    # --- 5. Define Metrics ---
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Ensure logits and labels are on CPU for numpy conversion if they aren't already
        # (Trainer usually handles this, but being explicit doesn't hurt)
        logits = logits if isinstance(logits, np.ndarray) else logits.cpu().numpy()
        labels = labels if isinstance(labels, np.ndarray) else labels.cpu().numpy()
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # --- 6. Define Dummy Training Arguments (needed for Trainer) ---
    # We only need arguments relevant to evaluation
    eval_output_dir = os.path.join(config.OUTPUT_DIR_BASE, "eval_temp")
    print(f"Using temporary directory for evaluation artifacts: {eval_output_dir}")

    eval_args = TrainingArguments(
        output_dir=eval_output_dir, # Needs an output directory
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        do_train=False, # Explicitly set do_train to False
        do_eval=True,   # Explicitly set do_eval to True
        fp16=torch.cuda.is_available(), # Match training setting if possible
        seed=config.SEED,
        report_to="none" # Disable logging integrations like W&B/TensorBoard for pure eval
    )

    # --- 7. Initialize Trainer (for evaluation) ---
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer, # Pass tokenizer for potential padding/collation
        compute_metrics=compute_metrics,
        # data_collator=None # Use default data collator
    )

    # --- 8. Run Evaluation ---
    print("Starting evaluation...")
    results = trainer.evaluate() # Specify dataset if different from eval_dataset if needed

    print(f"\n--- Evaluation Results for {model_path} on {dataset_split} split ---")
    for key, value in results.items():
        # Ensure value is float before formatting, handle potential non-numeric results gracefully
        if isinstance(value, (int, float)):
             print(f"{key}: {value:.4f}")
        else:
             print(f"{key}: {value}")


    # Clean up temporary directory if desired (optional)
    # import shutil
    # if os.path.exists(eval_output_dir):
    #     print(f"Removing temporary evaluation directory: {eval_output_dir}")
    #     shutil.rmtree(eval_output_dir)

    return results

if __name__ == "__main__":
    # Define the path to the model you want to evaluate
    # This path construction should match the output of your train.py script
    saved_model_path = os.path.join(config.OUTPUT_DIR_BASE, f"{config.MODEL_NAME}-finetuned-{config.DATASET_NAME}", "best_model")

    # Check if the saved model path exists before running
    if os.path.exists(saved_model_path) and os.path.isdir(saved_model_path):
        try:
            evaluate_model(model_path=saved_model_path, dataset_split="test")
        except Exception as e:
            print(f"\nAn error occurred during evaluation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Error: Saved model directory not found at '{saved_model_path}'")
        print("Please ensure training was completed successfully, the model was downloaded/transferred correctly,")
        print("and the path defined in src/evaluate.py matches the location.")