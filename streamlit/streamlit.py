import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import requests
from docx import Document  # Import the library to handle .docx files
import PyPDF2  # Import the library to handle PDF files

# Define colors and styles
st.markdown(
    """
    <style>
    .big-font {
        font-family:Helvetica; 
        font-size:50px !important; 
        font-weight: bold;
        color: #1E88E5;
    }
    .pred-font {
        font-family:Helvetica; 
        color: #FF4B4B;
        font-size:24px !important; 
    }
    .sidebar-style {
        background-color: #FDBA74;  /* Light orange background */
        padding: 10px;
    }
    .header-style {
        background-color: #FDBA74;  /* Light orange background */
        padding: 20px;
        border-radius: 0 0 10px 10px;
    }
    .result-box {
        background-color: #C8E6C9;  /* Light green background */
        padding: 20px;
        border-radius: 5px;
        margin-top: 10px;
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
st.sidebar.markdown('<div class="sidebar-style">', unsafe_allow_html=True)
st.sidebar.title('📚 Difficulty Level Predictor')
st.sidebar.subheader('This app allows you to predict the French difficulty level of a book. Never worry again about whether or not your French skills are sufficient to read a book. Use Bookly and find it out within seconds!')
st.sidebar.subheader('📄 Upload the Cover Text of your Book')

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("", type=["pdf", "docx"])
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<div class="header-style">', unsafe_allow_html=True)
st.markdown('<p class="big-font">Bookly</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        # Read as text file
        preface_text = str(uploaded_file.read(), 'utf-8')
    elif uploaded_file.type == "application/pdf":
        # Read as PDF file
        with st.spinner('📄 Extracting text from PDF...'):
            pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
            preface_text = ""
            for page_num in range(pdf_reader.numPages):
                preface_text += pdf_reader.getPage(page_num).extractText()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Read as DOCX file
        doc = Document(uploaded_file)
        preface_text = "\n".join([para.text for para in doc.paragraphs])
    
    st.write("### Uploaded Preface")
    st.text_area("", preface_text, height=250, help="This is the preface text extracted from your document.")

    # Progress bar during prediction
    with st.spinner('🔍 Predicting difficulty level...'):
        preface_transformed = vectorizer.transform([preface_text])
        prediction = model_LR.predict(preface_transformed)

    st.subheader('Predicted Difficulty Level')
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(f"<p class='pred-font'>{prediction[0]} 💡</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.sidebar.warning('Please upload a text, PDF, or Word document.')
