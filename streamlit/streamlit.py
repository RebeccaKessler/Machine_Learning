import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import requests
from charset_normalizer import from_bytes

url = 'https://github.com/RebeccaKessler/Machine_Learning/blob/main/streamlit/model_LR.pkl?raw=true'

def load_model(url):
    response = requests.get(url)
    data = pickle.loads(response.content)
    return data

# Load model and vectorizer
model_LR, vectorizer = load_model(url)

st.title('Book Difficulty Prediction App')
st.header('Upload the Preface of a Book')

# File uploader allows user to add their own preface
uploaded_file = st.file_uploader("Choose a file", type=["txt"])
if uploaded_file is not None:
    file_content = uploaded_file.read()
    
    # Attempt to detect the encoding
    try:
        results = from_bytes(file_content)  # Analyze the bytes to detect encoding
        preface_text = results.best().text  # Get the best match and extract text
        st.write("Uploaded Preface:")
        st.write(preface_text)

        # Predicting the difficulty
        preface_transformed = vectorizer.transform([preface_text])
        prediction = model_LR.predict(preface_transformed)

        st.subheader('Predicted Difficulty Level')
        st.write(prediction[0])
    except Exception as e:
        st.error(f"Failed to decode the file: {str(e)}")
        st.stop()
else:
    st.warning('Please upload a text file.')
