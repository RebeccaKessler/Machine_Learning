import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import requests
from docx import Document  # Import the library to handle .docx files

# Define colors and styles
st.markdown(
    """
    <style>
    .big-font {
        font-family: 'Arial Black', sans-serif; 
        font-size:30px !important; 
        font-weight: bold;
        color: #1E88E5;
    }
    .pred-font {
        font-family:Helvetica; 
        color: #FF4B4B;
        font-size:20px !important; 
    }
    .sidebar-style {
        background-color: #FDBA74;  /* Light orange background */
        padding: 20px;
        border-radius: 0px 15px 15px 0px;
    }
    .header-style {
        background-color: #E8EAF6;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True
)

# Load model and vectorizer once when the app starts
@st.cache(allow_output_mutation=True)
def load_model(url):
    response = requests.get(url)
    data = pickle.loads(response.content)
    return data

url = 'https://github.com/RebeccaKessler/Machine_Learning/blob/main/streamlit/model_LR.pkl?raw=true'
model_LR, vectorizer = load_model(url)

# Sidebar
st.markdown('<div class="sidebar-style">', unsafe_allow_html=True)
st.sidebar.title('📚 Difficulty Level Predictor')
st.sidebar.subheader('📄 Upload a Book Preface')
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<div class="header-style">', unsafe_allow_html=True)
st.markdown('<p class="big-font">Analyze Book Preface</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        # Read as text file
        preface_text = str(uploaded_file.read(), 'utf-8')
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Read as docx file
        doc = Document(uploaded_file)
        preface_text = "\n".join([para.text for para in doc.paragraphs])
    
    st.write("### Uploaded Preface")
    st.text_area("", preface_text, height=250, help="This is the preface text extracted from your document.")

    # Progress bar during prediction
    with st.spinner('🔍 Predicting difficulty level...'):
        preface_transformed = vectorizer.transform([preface_text])
        prediction = model_LR.predict(preface_transformed)

    st.subheader('Predicted Difficulty Level')
    st.write(f"{prediction[0]} 🔮")
else:
    st.sidebar.warning('Please upload a text or Word document.')
