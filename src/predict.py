# src/predict.py

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import config  
from data_loader import clean_html 



MODEL_PATH = os.path.join(
    config.OUTPUT_DIR_BASE,
    f"{config.MODEL_NAME}-finetuned-{config.DATASET_NAME}",
    "best_model"
)
DEVICE = config.DEVICE 
MAX_LENGTH = config.MAX_LENGTH

ID2LABEL = {0: "Negative", 1: "Positive"}

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

def predict_sentiment(text, tokenizer, model):
    """Predicts sentiment for a given text string."""
    if not text or not isinstance(text, str):
        return "Invalid input", None

    cleaned_text = clean_html(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length", 
        truncation=True
    )

    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

    with torch.no_grad(): 
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    predicted_class_id = torch.argmax(logits, dim=-1).item()

    predicted_label = ID2LABEL.get(predicted_class_id, "Unknown")

    probs_dict = {ID2LABEL.get(i, f"Class_{i}"): prob for i, prob in enumerate(probabilities)}

    return predicted_label, probs_dict

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