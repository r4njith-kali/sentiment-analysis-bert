import streamlit as st
st.set_page_config(page_title = "Sentiment Analyser", layout = "centered")

from src.predict import load_model_components, predict_sentiment, MODEL_PATH, ID2LABEL
from src import config
#import src.config as config


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return load_model_components(MODEL_PATH)

tokenizer, model = load_model()

#st.set_page_config(page_title = "Sentiment Analyser", layout = "centered")
st.title("Sentiment Analyzer using BERT")
st.markdown("Enter a review or a comment and get its Sentiment Analyzed.")

user_input = st.text_area("Enter your text here:", height = 200)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyse")
    elif tokenizer is None or model is None:
        st.error("Model is not loaded. Please check logs.")
    else:
        with st.spinner("Analysing..."):
            prediction, probs = predict_sentiment(user_input,tokenizer, model)
            st.success(f'Prediction: {prediction}')

            st.subheader("Confidence scores")
            st.bar_chart(probs)


