import streamlit as st
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import requests
from docx import Document  
import PyPDF2  

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
        font-family:Oregon; 
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
            df = pd.DataFrame(data, columns=["ID","Title", "Prediction"])
            st.write(df)
        else:
            st.write("No data found based on filter criteria.")

def display_library():
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
st.markdown('<p class="big-font">BOOKLY</p>', unsafe_allow_html=True)
st.write("### This app allows you to predict the French difficulty level of a book. Never worry again about whether or not your French skills are sufficient to read a book. Use Bookly and find it out within seconds!")

# Sidebar
with st.sidebar:
    st.write("## 📓 Upload the Abstract of your Book")
    title = st.text_input(" 🖊️ Enter the title of your book", key="book_title", help="Enter title of book.")
    uploaded_file = st.file_uploader("📄 Upload your abstract", type=["pdf", "docx"], help="Upload abstract of book.")
    predict_button = st.button("Predict Difficulty of Book")

    display_button = st.button("Display Library")
    if display_button:
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
            display_library()
   
#run model for prediction
if predict_button and uploaded_file is not None and title:
    if uploaded_file.type == "application/pdf":
        with st.spinner('📄 Extracting text from PDF...'):
            pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
            preface_text = ""
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                preface_text += page.extractText()
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
    save_to_library(title, prediction[0])

if 'filter_type' in st.session_state:
    display_library()



