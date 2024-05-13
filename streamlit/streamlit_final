from transformers import CamembertTokenizer, CamembertForSequenceClassification
import requests
from io import BytesIO
import tarfile

@st.cache(allow_output_mutation=True)
def download_and_load_model(model_url, tokenizer_url):
    # Model
    response = requests.get(model_url)
    model_file = BytesIO(response.content)
    model_dir = tarfile.open(fileobj=model_file)
    model_dir.extractall(path="./model")
    model = CamembertForSequenceClassification.from_pretrained('./model')

    # Tokenizer
    response = requests.get(tokenizer_url)
    tokenizer_file = BytesIO(response.content)
    tokenizer_dir = tarfile.open(fileobj=tokenizer_file)
    tokenizer_dir.extractall(path="./tokenizer")
    tokenizer = CamembertTokenizer.from_pretrained('./tokenizer')
    
    return model, tokenizer

# URLs to the model and tokenizer
model_url = 'https://github.com/RebeccaKessler/Machine_Learning/blob/main/streamlit/model.tar.gz?raw=true'
tokenizer_url = 'https://github.com/RebeccaKessler/Machine_Learning/blob/main/streamlit/tokenizer.tar.gz?raw=true'

# Load model and tokenizer
model, tokenizer = download_and_load_model(model_url, tokenizer_url)


if predict_button and uploaded_file is not None and title:
    preface_text = extract_text_from_uploaded_file(uploaded_file)  # Assumes you have a function to extract text

    # Encode and create a tensor for the input
    inputs = tokenizer(preface_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Predict difficulty level using the model
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_index = predictions.argmax().item()
        prediction_label = model.config.id2label[predicted_index]  # Assumes you have labels mapped to indices in the model config

    # Display the prediction result
    st.subheader('💡 Predicted Difficulty Level')
    st.success(f"{prediction_label}")

    # Automatically save prediction
    save_to_library(title, prediction_label)


    def extract_text_from_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(uploaded_file)
    return text
