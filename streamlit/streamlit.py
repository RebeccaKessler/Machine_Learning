import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import requests
from docx import Document  # Import the library to handle .docx files

# Load model and vectorizer once when the app starts
@st.cache(allow_output_mutation=True)
def load_model(url):
    response = requests.get(url)
    data = pickle.loads(response.content)
    return data

url = 'https://github.com/RebeccaKessler/Machine_Learning/blob/main/streamlit/model_LR.pkl?raw=true'
model_LR, vectorizer = load_model(url)

st.sidebar.title('Book Difficulty Prediction App')
st.sidebar.subheader('Upload the Preface of a Book')

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "docx"])
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

    # Progress bar during prediction
    with st.spinner('Predicting difficulty level...'):
        preface_transformed = vectorizer.transform([preface_text])
        prediction = model_LR.predict(preface_transformed)

    st.subheader('Predicted Difficulty Level')
    st.success(f"The predicted difficulty level is: {prediction[0]}")
else:
    st.sidebar.warning('Please upload a text or Word document.')
