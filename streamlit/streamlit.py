import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import requests
from docx import Document  # Import the library to handle .docx files
import PyPDF2  # Import the library to handle PDF files

# Simulate user login
if 'username' not in st.session_state:
    username = st.sidebar.text_input("Username")
    if st.sidebar.button("Login"):
        st.session_state.username = username

import sqlite3

# Initialize connection
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Create table
c.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        username TEXT,
        preface TEXT,
        prediction TEXT
    )
''')
conn.commit()

# Function to save data
def save_data(username, preface, prediction):
    c.execute('INSERT INTO user_data (username, preface, prediction) VALUES (?, ?, ?)', (username, preface, prediction))
    conn.commit()

# Function to load user data
def load_data(username):
    c.execute('SELECT preface, prediction FROM user_data WHERE username = ?', (username,))
    return c.fetchall()    

# Define colors and styles
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://github.com/RebeccaKessler/Machine_Learning/blob/main/streamlit/background.png?raw=true");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .big-font {
        font-family:Helvetica; 
        font-size:100px !important;
        font-weight: bold;
        color: #FDBA74; 
    }
    .pred-font {
        font-family:Helvetica; 
        color: #000000;
        font-size:24px !important;
    }
    .header-style {
        padding: 20px;
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
st.sidebar.title('Difficulty Level Predictor')
st.sidebar.subheader('📄 Upload the Cover Text of your Book')

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("", type=["pdf", "docx"])
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<div class="header-style">', unsafe_allow_html=True)
st.markdown('<p class="big-font">📚 Bookly</p>', unsafe_allow_html=True)
st.subheader('This app allows you to predict the French difficulty level of a book. Never worry again about whether or not your French skills are sufficient to read a book. Use Bookly and find it out within seconds!')
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
    
    st.write("### 📄 Uploaded Preface")
    st.text_area("", preface_text, height=250, help="This is the preface text extracted from your document.")

    # Progress bar during prediction
    with st.spinner('🔍 Predicting difficulty level...'):
        preface_transformed = vectorizer.transform([preface_text])
        prediction = model_LR.predict(preface_transformed)

    st.subheader('💡 Predicted Difficulty Level')
    st.markdown('<div class="result-box"><p class="pred-font">' + prediction[0] + '</p></div>', unsafe_allow_html=True)
else:
    st.sidebar.warning('Please upload a PDF or Word document.')


if 'username' in st.session_state:
    # Add save functionality within the file processing section
    if uploaded_file is not None:
        # Existing file processing logic...
        # Save to database
        if st.button("Save to Profile"):
            save_data(st.session_state.username, preface_text, prediction)
            st.success("Saved!")

    # Add functionality to show saved entries
    if st.sidebar.button("Show My Library"):
        user_data = load_data(st.session_state.username)
        for preface, prediction in user_data:
            st.write("Preface:", preface)
            st.write("Prediction:", prediction)
