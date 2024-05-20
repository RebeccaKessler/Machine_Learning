import streamlit as st
import sqlite3
import pandas as pd
import requests
import tarfile
from io import BytesIO
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
from PyPDF2 import PdfFileReader
from docx import Document

# Define colors and styles
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://github.com/RebeccaKessler/Machine_Learning/blob/main/streamlit/background_final.png?raw=true");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .big-font {
        font-family:Helvetica; 
        font-size:140px !important;
        font-weight: bold;
        color: #000000; 
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

import streamlit as st
import sqlite3
import pandas as pd
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from PyPDF2 import PdfFileReader
from docx import Document
import requests
import os

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
        c.execute("INSERT INTO library (title, prediction) VALUES (?, ?)", (title, prediction))
        conn.commit()

def fetch_and_display_library(query, params):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute(query, params)
        data = c.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["ID", "Title", "Prediction"])
            df = df[["Title", "Prediction"]]
            st.write(df)
        else:
            st.write("No data found based on filter criteria.")

def display_library():
    st.write("## 📓 Library")
    if 'filter_type' in st.session_state and st.session_state.filter_type in ["Title", "Prediction Level"]:
        filter_type = st.session_state.filter_type
        filter_value = st.session_state.filter_value

        if filter_type == "Title" and filter_value:
            query = "SELECT * FROM library WHERE title LIKE ?"
            params = ('%' + filter_value + '%',)
        elif filter_type == "Prediction Level" and filter_value:
            query = "SELECT * FROM library WHERE prediction = ?"
            params = (filter_value,)
    else:
        query = "SELECT * FROM library"
        params = ()

    fetch_and_display_library(query, params)

# Function to download files from a URL
def download_file(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)
        
# Load model and tokenizer once when the app starts
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(model_url, tokenizer_url):
    # Paths where the files will be saved
    model_dir = "camembert_model"
    tokenizer_dir = "camembert_tokenizer"

    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    # Helper function to download files
    def download_files(urls, save_dir):
        for url in urls:
            file_name = os.path.basename(url)
            save_path = os.path.join(save_dir, file_name)
            download_file(url, save_path)
    
    # Download model files
    model_files = [f"{model_url}/pytorch_model.bin", f"{model_url}/config.json"]
    download_files(model_files, model_dir)

    # Download tokenizer files
    tokenizer_files = [f"{tokenizer_url}/tokenizer.json", f"{tokenizer_url}/tokenizer_config.json", f"{tokenizer_url}/special_tokens_map.json", f"{tokenizer_url}/vocab.json", f"{tokenizer_url}/merges.txt"]
    download_files(tokenizer_files, tokenizer_dir)
    
    # Load model and tokenizer
    model = CamembertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = CamembertTokenizer.from_pretrained(tokenizer_dir)
    
    return model, tokenizer

# URLs for the model and tokenizer (replace with your actual URLs)
model_url = 'https://raw.githubusercontent.com/YourUsername/YourRepo/main/path/to/saved/model'
tokenizer_url = 'https://raw.githubusercontent.com/RebeccaKessler/Machine_Learning/main/streamlit/tokenizer_config.json'

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(model_url, tokenizer_url)


def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfFileReader(uploaded_file)
    preface_text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        preface_text += page.extractText()
    return preface_text

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    preface_text = "\n".join([para.text for para in doc.paragraphs])
    return preface_text

# Main content
st.markdown('<p class="big-font">BOOKLY</p>', unsafe_allow_html=True)
st.write("### This app allows you to predict the French difficulty level of a book. Never worry again about whether or not your French skills are sufficient to read a book. Use Bookly and find it out within seconds!")

# Sidebar
with st.sidebar:
    st.write("# 📓 Upload Excerpt of your Book")
    st.markdown("##")
    title = st.text_input("🖊️ Enter the title of your book", key="book_title", help="Enter title of book.")
    uploaded_file = st.file_uploader("📄 Upload your excerpt", type=["pdf", "docx"], help="Upload abstract of book.")
    predict_button = st.button("Predict Difficulty of Book")
    st.markdown("##") 
    display_button = st.button("Display Library")
    if display_button or 'show_filters' in st.session_state:
        st.session_state.show_filters = True

    if 'show_filters' in st.session_state and st.session_state.show_filters:
        filter_options = st.radio("Filter by:", ["Title", "Prediction Level"], index=0, key='filter_selection')
        if filter_options == "Title":
            title_filter = st.text_input("Enter Title:", key='title_filter')
        elif filter_options == "Prediction Level":
            pred_filter = st.selectbox("Select Prediction Level", ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], key='pred_filter')
        
        if st.button("Apply Filters"):
            st.session_state.filter_type = filter_options
            st.session_state.filter_value = st.session_state.title_filter if filter_options == "Title" else st.session_state.pred_filter if filter_options == "Prediction Level" else None
            st.session_state['display_library'] = True

if predict_button and uploaded_file is not None and title:
    preface_text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else extract_text_from_docx(uploaded_file)
    
    inputs = tokenizer(preface_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_index = predictions.argmax().item()
    prediction_label = model.config.id2label[predicted_index]

    st.subheader('💡 Predicted Difficulty Level')
    st.success(f"{prediction_label}")

    save_to_library(title, prediction_label)

if 'filter_type' in st.session_state:
    display_library()



 
