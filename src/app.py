import streamlit as st
from predict import load_model_components, predict_sentiment
from predict import MODEL_PATH
import config

MODEL_PATH = predict.MODEL_PATH
DEVICE = config.DEVICE
MAX_LENGTH = config.MAX_LENGTH

ID2LABEL = predict.ID2LABEL

tokenizer, model = load_model_components(MODEL_PATH)

st.set_page_config(page_title = "Sentiment Analyser", layout = "centered")