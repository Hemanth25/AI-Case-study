# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 23:07:19 2025

@author: hemanthn
"""


import streamlit as st
from logger import setup_logger
from config import DATA_PATH
from model_utils import train_model
from document_utils import extract_text_from_pdf, extract_text_from_image
from tools_and_graph import create_graph
from chromadb import Client
from chromadb.utils import embedding_functions
from config import KEY

logger = setup_logger()

#Setting up chroma DB
chroma_client = Client()
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=KEY, model_name="text-embedding-ada-002")
collection = chroma_client.get_or_create_collection(name="applicant_doc", embedding_function=embedding_fn)
logger.info("ChromaDB collection created.")
def store_in_chroma(text, source):
    collection.add(documents=[text], metadatas=[{"source": source}], ids=[source])
    logger.info(f"Stored text in ChromaDB under source: {source}")

st.title("Interactive Social Support Chatbot")
model, scaler = train_model(DATA_PATH)
# Saving Data in DB
uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else extract_text_from_image(uploaded_file)
    store_in_chroma(text, uploaded_file.name)
    st.session_state["messages"].append(("assistant", "Document processed and stored in ChromaDB."))
# creating Graph
graph = create_graph(model, scaler, collection, logger)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

prompt = """Please Share your details as mentioned below to check the financial eligibility

1. monthly income:
2. Employment years:
3. Household Members:
4. Wealth index:


"""
st.session_state["messages"].append(("system", prompt))
response = graph.invoke({"messages": prompt})


user_input = st.chat_input("Ask about eligibility, support or documents")
if user_input:
    st.session_state["messages"].append(("user", user_input))
    response = graph.invoke({"messages": st.session_state["messages"]})
    reply = response["messages"][-1].content
    st.session_state["messages"].append(("assistant", reply))

for role, message in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(message)