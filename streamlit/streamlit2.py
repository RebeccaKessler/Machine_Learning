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

def save_to_library(title, prediction):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO library (title, prediction)
            VALUES (?, ?)
            ''', (title, prediction))
        conn.commit()

ef show_library():
    filter_type = st.sidebar.selectbox("Filter by", ["None", "Title", "Prediction"])
    if filter_type != "None":
        filter_value = st.sidebar.text_input(f"Enter {filter_type}")
        if filter_value:
            with sqlite3.connect(DB_FILE) as conn:
                c = conn.cursor()
                c.execute(f"SELECT * FROM library WHERE {filter_type.lower()} LIKE ?", ('%' + filter_value + '%',))
                data = c.fetchall()
    else:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM library")
            data = c.fetchall()

    if data:
        df = pd.DataFrame(data, columns=["ID", "Title", "Prediction"])
        st.table(df)
    else:
        st.write("No data found based on filter criteria.")


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
st.markdown('<p class="big-font">Bookly</p>', unsafe_allow_html=True)
st.write("### This app allows you to predict the French difficulty level of a book. Never worry again about whether or not your French skills are sufficient to read a book. Use Bookly and find it out within seconds!")

# Sidebar
with st.sidebar:
    st.write("### Upload the Abstract of your Book")
    title = st.text_input("Enter the title of your book", key="book_title", help="Enter the title of your book here.")
    uploaded_file = st.file_uploader("", type=["pdf", "docx"], help="Upload the abstract of your book here.")

#run model for prediction
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        with st.spinner('📄 Extracting text from PDF...'):
            pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
            preface_text = ""
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        preface_text = "\n".join([para.text for para in doc.paragraphs])

    #print result of prediction
    st.markdown("##") 
    st.write("### 📄 Uploaded Preface")
    st.text_area("", preface_text, height=250, help="This is the preface text extracted from your document.")

    # Progress bar during prediction
    with st.spinner('🔍 Predicting difficulty level...'):
        preface_transformed = vectorizer.transform([preface_text])
        prediction = model_LR.predict(preface_transformed)

    st.subheader('💡 Predicted Difficulty Level')
    st.success(f"{prediction[0]}")

    #Automatically save prediction
    if title:
        save_to_library(title, prediction[0])

# Library view button
with st.sidebar:
    if st.button('Show Library', key='library_button'):
        show_library()

    
