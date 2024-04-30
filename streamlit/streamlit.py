import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

url = 'https://github.com/RebeccaKessler/Machine_Learning/blob/main/streamlit/model_LR.pkl'

def load_model(url):
    response = requests.get(url)
    data = pickle.loads(response.content)
    return data

#load model and vectorizer
model_LR, vectorizer = load_model()

st.title('Book Difficulty Prediction App')
st.header('Upload the Preface of a Book')

# File uploader allows user to add their own preface
uploaded_file = st.file_uploader("Choose a file", type=["txt"])
if uploaded_file is not None:
    # To read file as string:
    preface_text = str(uploaded_file.read(), 'utf-8')
    st.write("Uploaded Preface:")
    st.write(preface_text)

    # Predicting the difficulty
    preface_transformed = vectorizer.transform([preface_text])
    prediction = model_LR.predict(preface_transformed)

    st.subheader('Predicted Difficulty Level')
    st.write(prediction[0])
else:
    st.warning('Please upload a text file.')
    st.stop()
