🧠 Interactive Social Support Chatbot
This project is an intelligent, multimodal chatbot built using LangGraph, OpenAI GPT-4, and a locally trained XGBoost ML model. It automates decision-making for economic and social support applications, offering real-time recommendations and eligibility assessments based on structured and unstructured data.

🚀 Features
- ✅ ML-based Eligibility Assessment
Predicts financial support eligibility using income, employment history, family size, and wealth index.
- 🤖 Agentic AI Orchestration with LangGraph
Uses LangGraph's ReAct-style agent to reason and route tasks across multiple tools.
- 🧾 Multimodal Document Processing
Extracts and stores text from uploaded PDFs and images using OCR and ChromaDB.
- 💬 Interactive Chat Interface
Built with Streamlit for real-time user interaction and document-based Q&A.
- 📚 Semantic Search with ChromaDB
Embeds and retrieves applicant documents using OpenAI embeddings.

🛠️ Tech Stack
| Component | Technology | 
| ML Model | XGBoost (local) | 
| LLM | OpenAI GPT-4 | 
| Agent Framework | LangGraph | 
| Embeddings | OpenAI text-embedding-ada-002 | 
| Vector DB | ChromaDB | 
| OCR | Tesseract via pytesseract | 
| PDF Parsing | PyMuPDF (fitz) | 
| UI | Streamlit | 



📦 Setup Instructions
1. Clone the Repository


2. Install Dependencies
pip install -r requirements.txt


Make sure you have Tesseract installed and configured: Download Tesseract OCR

3. Set Your OpenAI API Key
Edit the script or set it as an environment variable:
os.environ["OPENAI_API_KEY"] = "key"

4. Set Your "tesseract.exe" path (if not installed, please install locally)
pytesseract.pytesseract.tesseract_cmd = "tesseract.exe path"

📊 Data Requirements
Place your input CSV file at:
C:/Users/hemanthn/OneDrive - Nagarro/Documents/Projects/Abu/Data/Input.csv


The CSV should include the following columns:
- income
- employment_years
- family_size
- wealth_index
- eligible (target label)

🧪 How It Works
- Train ML Model: Uses XGBoost to classify eligibility.
- Process Documents: Extracts text from PDFs/images and stores in ChromaDB.
- Agent Reasoning: LangGraph agent uses tools to:
- Format raw text for ML
- Check eligibility
- Recommend support or enablement
- Search stored documents
- Chat Interface: Users interact via Streamlit to ask questions or upload files.

💬 Usage
Run the chatbot:
streamlit run your_script.py


Upload a document and ask questions like:
- “Am I eligible for financial support?”
- “Suggest training programs for me.”
- “What does my uploaded document say?”

🧰 Tools Defined
- eligibility_checker: Uses ML model to assess eligibility.
- social_support_recommendation: GPT-4-based approval advice.
- economic_enablement_recommendation: GPT-4-based enablement suggestions.
- format_data_for_ml: Parses raw text into ML-ready format.
- query_documents: Searches stored documents in ChromaDB.

📂 File Upload Support
- Image (.png, .jpg)
Uploaded files are processed and embedded for semantic search
