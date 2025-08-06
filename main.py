# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 16:20:10 2025

@author: hemanthn
"""

key=""

import os
os.environ["OPENAI_API_KEY"] = key
#%%
import pandas as pd
# import numpy as np
import fitz  # PyMuPDF for PDF text extraction
from PIL import Image
import pytesseract
from chromadb import Client
from chromadb.utils import embedding_functions
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
import streamlit as st
from sklearn.model_selection import train_test_split

# ================== TRAIN ML MODEL ==================
data = pd.read_csv("C:/Users/hemanthn/OneDrive - Nagarro/Documents/Projects/Abu/Data/Input.csv")
X = data[['income', 'employment_years', 'family_size', 'wealth_index']]
y = data['eligible']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = xgb.XGBClassifier(learning_rate=0.04, max_depth=4, reg_alpha=0.4)
model.fit(X_train_scaled, y_train)

# ================== CHROMADB SETUP ==================

chroma_client = Client()
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=key, model_name="text-embedding-ada-002")
collection = chroma_client.get_or_create_collection(name="applicant_doc", embedding_function=embedding_fn)

def store_in_chroma(text, source):
    collection.add(documents=[text], metadatas=[{"source": source}], ids=[source])

def query_chroma(query_text):
    results = collection.query(query_texts=[query_text], n_results=3)
    return results

# ================== DOCUMENT PROCESSING ==================
def extract_text_from_pdf(file_path):
    text = ""
    pdf = fitz.open(file_path)
    for page in pdf:
        text += page.get_text()
    return text

def extract_text_from_image(image_path):
    pytesseract.pytesseract.tesseract_cmd = "C:/Users/hemanthn/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

# ================== defining TOOLS ==================
def check_eligibility_tool(input_str):
    try:
        income, employment_years, family_size, wealth_index = map(int, input_str.split(","))
        features = scaler.transform([[income, employment_years, family_size, wealth_index]])
        return "Eligible" if model.predict(features)[0] == 1 else "Not Eligible"
    except:
        return "Invalid input. Please provide: income, employment_years, family_size, wealth_index"

llm = ChatOpenAI(model="gpt-4.1", temperature=0)

def social_support_recommendation_tool(query):
    return llm.invoke(f"Applicant status: {query}. Approve or soft decline financial support and explain why.").content

def economic_enablement_recommendation_tool(query):
    return llm.invoke(f"Suggest upskilling, job matching, and career counseling for: {query}").content

def format_data_for_ml_tool(raw_text):
    prompt = f"""
    Extract numerical values from the following text for ML model input:
    1. Income
    2. Employment Years
    3. Family Size
    4. Wealth Index
    
    Return as: income, employment_years, family_size, wealth_index
    
    Text: {raw_text}
    """
    return llm.invoke(prompt).content

# Wrap tools for LangGraph
tools = [
    Tool(name="eligibility_checker", func=check_eligibility_tool, description="Check eligibility based on numerical inputs."),
    Tool(name="social_support_recommendation", func=social_support_recommendation_tool, description="Approve or soft decline financial support."),
    Tool(name="economic_enablement_recommendation", func=economic_enablement_recommendation_tool, description="Suggest training, job matching, and career counseling."),
    Tool(name="format_data_for_ml", func=format_data_for_ml_tool, description="Parse unstructured text into ML model input format."),
    Tool(name="query_documents", func=lambda x: query_chroma(x), description="Search applicant documents stored in ChromaDB.")
]

# Create ReAct Agent with LangGraph
agent = create_react_agent(model=llm, tools=tools)
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.set_entry_point("agent")
app = graph.compile()

# ================== 5. STREAMLIT INTERACTIVE CHAT ==================
st.title("Interactive Social Support Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        extracted_text = extract_text_from_image(uploaded_file)
    else:
        extracted_text = extract_text_from_image(uploaded_file)
    store_in_chroma(extracted_text, uploaded_file.name)
    st.session_state["messages"].append(("assistant", "Document processed and stored in ChromaDB."))

user_query = st.chat_input("Ask anything about eligibility, recommendations, or uploaded documents")
if user_query:
    st.session_state["messages"].append(("user", user_query))
    result = app.invoke({"messages": st.session_state["messages"]})
    bot_reply = result["messages"][-1].content
    st.session_state["messages"].append(("assistant", bot_reply))

# Display chat history
for role, message in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(message)
