# src/evaluate.py

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

def evaluate_model(model_path, dataset_split="test"):
    """Loads a fine-tuned model and evaluates it on a specified dataset split."""

    print(f"--- Evaluating Model from: {model_path} ---")
    print(f"Using dataset split: {dataset_split}")

    print(f"Setting random seed to: {config.SEED}")
    set_seed(config.SEED)

    print("Loading tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Warning: Error loading tokenizer from {model_path}. Trying base model {config.MODEL_NAME}. Error: {e}")
        print("This might happen if tokenizer files weren't saved correctly with the model.")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)


    print("Loading and preprocessing data...")
    tokenized_datasets = load_and_preprocess_data(
        tokenizer_name=model_path, 
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

    num_labels = len(eval_dataset.unique("labels")) 
    print(f"Loading fine-tuned model from {model_path} with {num_labels} labels.")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels
    )
    effective_device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Moving model to device: {effective_device}")
    model.to(effective_device)

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits if isinstance(logits, np.ndarray) else logits.cpu().numpy()
        labels = labels if isinstance(labels, np.ndarray) else labels.cpu().numpy()
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # --- Define Dummy Training Arguments (needed for Trainer) ---
    # We only need arguments relevant to evaluation
    eval_output_dir = os.path.join(config.OUTPUT_DIR_BASE, "eval_temp")
    print(f"Using temporary directory for evaluation artifacts: {eval_output_dir}")

    eval_args = TrainingArguments(
        output_dir=eval_output_dir, 
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        do_train=False, 
        do_eval=True,   
        fp16=torch.cuda.is_available(), 
        seed=config.SEED,
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics,
    )

    print("Starting evaluation...")
    results = trainer.evaluate() 

    print(f"\n--- Evaluation Results for {model_path} on {dataset_split} split ---")
    for key, value in results.items():
        if isinstance(value, (int, float)):
             print(f"{key}: {value:.4f}")
        else:
             print(f"{key}: {value}")

    return results

if __name__ == "__main__":
    saved_model_path = os.path.join(config.OUTPUT_DIR_BASE, f"{config.MODEL_NAME}-finetuned-{config.DATASET_NAME}", "best_model")

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