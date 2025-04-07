# src/predict.py

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src import config  # Import your configuration
from src.data_loader import clean_html # Import the cleaning function

# --- Configuration ---
# Construct the path to the saved fine-tuned model
MODEL_PATH = os.path.join(
    config.OUTPUT_DIR_BASE,
    f"{config.MODEL_NAME}-finetuned-{config.DATASET_NAME}",
    "best_model"
)
DEVICE = config.DEVICE # Use device from config (checks for CUDA)
MAX_LENGTH = config.MAX_LENGTH # Use max length from config

# Map prediction index to label name
ID2LABEL = {0: "Negative", 1: "Positive"}

# --- Load Model and Tokenizer ---
def load_model_components(model_path):
    """Loads the fine-tuned model and tokenizer."""
    print(f"Loading model and tokenizer from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at {model_path}")
        print("Please ensure training is complete and the path is correct.")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE) # Move model to GPU if available, else CPU
        model.eval()     # Set model to evaluation mode (important!)
        print("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return None, None

# --- Prediction Function ---
def predict_sentiment(text, tokenizer, model):
    """Predicts sentiment for a given text string."""
    if not text or not isinstance(text, str):
        return "Invalid input", None

    # 1. Clean Text (optional, but good practice if model saw cleaned data)
    cleaned_text = clean_html(text) # Apply same cleaning as training

    # 2. Tokenize Text
    # return_tensors='pt' gets PyTorch tensors
    # padding=True, truncation=True ensures handling various lengths
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length", # Pad to max_length
        truncation=True
    )

    # 3. Move inputs to the correct device
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

    # 4. Make Prediction
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model(**inputs)

    # 5. Process Output
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0] # Get probabilities
    predicted_class_id = torch.argmax(logits, dim=-1).item() # Get predicted class index

    # 6. Map to Label
    predicted_label = ID2LABEL.get(predicted_class_id, "Unknown")

    # Create a dictionary of probabilities per label
    probs_dict = {ID2LABEL.get(i, f"Class_{i}"): prob for i, prob in enumerate(probabilities)}

    return predicted_label, probs_dict


# --- Main Execution Block ---
if __name__ == "__main__":
    tokenizer, model = load_model_components(MODEL_PATH)

    if tokenizer and model:
        print(f"\nModel ready. Using device: {DEVICE}")
        print('Enter text to analyze sentiment (or type "quit" to exit):')

        while True:
            user_input = input(">> ")
            if user_input.lower().strip() == "quit":
                break

            predicted_label, probabilities = predict_sentiment(user_input, tokenizer, model)

            print(f"   Prediction: {predicted_label}")
            if probabilities:
                 print(f"   Probabilities: {probabilities}")
            print("-" * 20)

    else:
        print("Could not load model. Exiting.")