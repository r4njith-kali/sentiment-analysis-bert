import streamlit as st
from predict import load_model_components, predict_sentiment
import config


def load_model():
