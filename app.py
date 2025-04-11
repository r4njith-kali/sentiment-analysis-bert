import streamlit as st
st.set_page_config(page_title = "Sentiment Analyser", layout = "centered")

from src.predict import load_model_components, predict_sentiment, MODEL_PATH, ID2LABEL
from src import config
#import src.config as config


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return load_model_components(MODEL_PATH)

tokenizer, model = load_model()

st.title("Sentiment Analyzer using BERT")
st.markdown("Enter the number of reviews you want to analyze:")

num_reviews = st.number_input("Number of reviews", min_value = 1, max_value = 10, value = 1, step = 1)


st.markdown("Enter a review or a comment and get its Sentiment Analyzed.")

user_inputs = []

for i in range(num_reviews):
    user_input = st.text_area(f"Review {i+1}", height = 100, key = f"review_{i}")
    user_inputs.append(user_input)

if st.button("Analyze Sentiment"):
    for idx, review in enumerate(user_inputs):
        if not review.strip():
            st.warning(f"Review {idx+1} is empty. Please enter text.")
        elif tokenizer is None or model is None:
            st.error("Model is not loaded. Please check logs")
        else:
            with st.spinner(f"Analyzing review {idx+1}..."):
                prediction, probs = predict_sentiment(review, tokenizer, model)
                st.subheader(f"Review {idx+1}:")
                st.success(f'Prediction: {prediction}')
                st.bar_chart(probs)

