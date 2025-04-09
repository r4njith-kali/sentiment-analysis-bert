# src/app.py

import os
from flask import Flask, render_template, request
import config
from predict import load_model_components, predict_sentiment

app = Flask(__name__)

# Construct the model path as defined in predict.py
MODEL_PATH = os.path.join(
    config.OUTPUT_DIR_BASE,
    f"{config.MODEL_NAME}-finetuned-{config.DATASET_NAME}",
    "best_model"
)

# Load the model and tokenizer once when the application starts
tokenizer, model = load_model_components(MODEL_PATH)

@app.route('/')
def index():
    """Render the main page with the text input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission: read text input from the user,
    make a sentiment prediction using the loaded model,
    and render the result.
    """
    # Get the text from the form
    text = request.form.get('text')
    
    # Use the predict_sentiment function from predict.py (&#8203;:contentReference[oaicite:2]{index=2})
    predicted_label, probabilities = predict_sentiment(text, tokenizer, model)
    
    # Render a template to display the result
    return render_template('result.html', text=text, label=predicted_label, probabilities=probabilities)

if __name__ == '__main__':
    # Run the app in debug mode for development purposes
    app.run(debug=True)
