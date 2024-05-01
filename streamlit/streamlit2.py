import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import requests
from docx import Document  # Import the library to handle .docx files
import PyPDF2  # Import the library to handle PDF files
import sqlite3

# Define colors and styles
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://github.com/RebeccaKessler/Machine_Learning/blob/main/streamlit/background1.png?raw=true");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .big-font {
        font-family:Helvetica; 
        font-size:100px !important;
        font-weight: bold;
        color: #000000; 
    }
    .pred-font {
        font-family:Helvetica; 
        color: #000000;
        font-size:24px !important;
    }
    .header-style {
        padding: 20px;
        background-color: #808080;
        border-radius: 0 0 10px 10px;
    }
    .result-box {
        background-color: #C8E6C9; 
        padding: 5px;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)
# Database setup
DB_FILE = "library.db"
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute('''
          CREATE TABLE IF NOT EXISTS library
          ([generated_id] INTEGER PRIMARY KEY, [title] text, [prediction] text)
          ''')
conn.commit()

# Load model and vectorizer once when the app starts
@st.cache(allow_output_mutation=True)
def load_model(url):
    response = requests.get(url)
    data = pickle.loads(response.content)
    return data

#import our final model
url = 'https://github.com/RebeccaKessler/Machine_Learning/blob/main/streamlit/model_LR.pkl?raw=true'
model_LR, vectorizer = load_model(url)

# Main content
st.markdown('<div class="header-style">', unsafe_allow_html=True)
st.markdown('<p class="big-font">📚 Bookly</p>', unsafe_allow_html=True)
st.subheader('This app allows you to predict the French difficulty level of a book.')
st.markdown('</div>', unsafe_allow_html=True)

#Sidebar
with st.sidebar:
    st.subheader('📄 Upload the Cover Text of your Book')
    title = st.text_input("Enter the book title", key="book_title")
    uploaded_file = st.file_uploader("", type=["pdf", "docx"])

# Function to save to database
def save_to_library(title, prediction):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
                  INSERT INTO library (title, prediction)
                  VALUES (?, ?)
                  ''', (title, prediction))
        conn.commit()

if st.sidebar.button('Library'):
    save_to_library(title, prediction[0])
    st.sidebar.success("Saved to Library!")


#run model for prediction
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
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

    #print result of prediction
    st.write("### 📄 Uploaded Preface")
    st.text_area("", preface_text, height=250, help="This is the preface text extracted from your document.")

    # Progress bar during prediction
    with st.spinner('🔍 Predicting difficulty level...'):
        preface_transformed = vectorizer.transform([preface_text])
        prediction = model_LR.predict(preface_transformed)

    st.subheader('💡 Predicted Difficulty Level')
    st.success(f"{prediction[0]}")

if st.sidebar.button('Library'):
        save_to_library(title, prediction[0])
        st.sidebar.success("Saved to Library!")

if st.sidebar.button('Library') and title and prediction:
    save_to_library(title, prediction[0])
    st.sidebar.success("Saved to Library!")
else:
    st.sidebar.error("Please enter a title and make a prediction before saving.")
    
