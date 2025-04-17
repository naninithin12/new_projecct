import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # Fix async event loop error

import streamlit as st
import json
import uuid
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import fitz  # PyMuPDF
import numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

# --- Constants ---
USER_DATA_FILE = 'users.json'
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'vgg_model.pth'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

food_categories = ["Fruits", "Grains", "Vegetables", "Proteins", "Dairy"]
vitamin_recommendations = {
    "Fruits": {"deficiency": "Vitamin A", "recommendation": "Eat carrots, sweet potatoes, and spinach."},
    "Vegetables": {"deficiency": "Vitamin C", "recommendation": "Consume bell peppers and citrus fruits."},
    "Grains": {"deficiency": "Vitamin B", "recommendation": "Eat whole grains like oats and quinoa."},
    "Proteins": {"deficiency": "Vitamin D", "recommendation": "Include fatty fish and egg yolks."},
    "Dairy": {"deficiency": "Vitamin E", "recommendation": "Consume almonds and sunflower seeds."}
}

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load Models ---
@st.cache_resource
def get_models():
    vgg_model = models.vgg16(pretrained=True)
    vgg_model.classifier[6] = nn.Linear(vgg_model.classifier[6].in_features, len(food_categories))
    vgg_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    vgg_model.eval()

    # ✅ Faster embedding model
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')  # ~45MB
    text_generator = pipeline("text-generation", model="tiiuae/falcon-rw-1b", device=0)  # If you have a small GPU


    return vgg_model, embedding_model, text_generator

model, embedding_model, generator = get_models()

# --- Session State ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'menu_option' not in st.session_state:
    st.session_state.menu_option = None

# --- Auth Utilities ---
def load_users():
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'w') as f:
            json.dump({}, f)
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# --- Auth Pages ---
def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users = load_users()
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
        else:
            st.error("Invalid credentials.")

def register_page():
    st.title("📝 Register")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    if st.button("Register"):
        users = load_users()
        if username in users:
            st.error("Username already exists.")
        else:
            users[username] = password
            save_users(users)
            st.success("Registration successful!")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.menu_option = None
    st.success("Logged out!")

# --- Food Detection Dashboard ---
def food_dashboard():
    st.title(f"🍽️ Welcome, {st.session_state.username}")
    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            _, pred_class = torch.max(output, 1)
        food = food_categories[pred_class.item()]
        vitamin_info = vitamin_recommendations[food]
        st.subheader(f"Predicted Food Category: {food}")
        st.markdown(f"**Deficiency:** {vitamin_info['deficiency']}")
        st.markdown(f"**Recommendation:** {vitamin_info['recommendation']}")

# --- Document QA System ---
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

@st.cache_data
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)

def get_file_hash(file_data):
    return hashlib.md5(file_data).hexdigest()

@st.cache_resource
def create_faiss_index_cached(chunks, file_hash):
    embeddings = embedding_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

def get_top_k_chunks(query, index, chunks, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

def answer_question(query, index, chunks):
    context = "\n".join(get_top_k_chunks(query, index, chunks))[:800]
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_new_tokens=100, do_sample=True, pad_token_id=50256)[0]['generated_text']
    return result.split("Answer:")[-1].strip()

def document_qa():
    st.title("📄 Document QA using RAG")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf:
        file_bytes = uploaded_pdf.read()
        file_hash = get_file_hash(file_bytes)
        uploaded_pdf.seek(0)
        text = extract_text_from_pdf(uploaded_pdf)
        if text:
            with st.spinner("Indexing your document..."):
                chunks = chunk_text(text)
                index, stored_chunks = create_faiss_index_cached(chunks, file_hash)
            st.success("PDF indexed. You can now ask questions.")
            query = st.text_input("Enter your question:")
            if query:
                answer = answer_question(query, index, stored_chunks)
                st.subheader("Answer:")
                st.write(answer)

# --- Main Menu ---
def main_menu():
    if st.session_state.logged_in:
        menu = st.sidebar.selectbox("📌 Menu", ["🏠 Food Dashboard", "📄 Document QA", "🔓 Logout"])
        if menu == "🏠 Food Dashboard":
            food_dashboard()
        elif menu == "📄 Document QA":
            document_qa()
        elif menu == "🔓 Logout":
            logout()
    else:
        menu = st.sidebar.selectbox("📌 Menu", ["🔐 Login", "📝 Register"])
        if menu == "🔐 Login":
            login_page()
        elif menu == "📝 Register":
            register_page()

# --- Run App ---
if __name__ == "__main__":
    main_menu()
