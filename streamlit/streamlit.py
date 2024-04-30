import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import requests
from docx import Document  # Import the library to handle .docx files

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
uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx"])
if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        # Read as text file
        preface_text = str(uploaded_file.read(), 'utf-8')
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Read as docx file
        doc = Document(uploaded_file)
        preface_text = "\n".join([para.text for para in doc.paragraphs])

    st.write("Uploaded Preface:")
    st.write(preface_text)

    # Predicting the difficulty
    preface_transformed = vectorizer.transform([preface_text])
    prediction = model_LR.predict(preface_transformed)

    st.subheader('Predicted Difficulty Level')
    st.write(prediction[0])
else:
    st.warning('Please upload a text or Word document.')
