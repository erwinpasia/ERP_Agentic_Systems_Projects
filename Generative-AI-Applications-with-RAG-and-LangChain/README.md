# 🤖 Unified AI Chatbot with RAG  
**Empowering Conversations with Document Intelligence**

---

## 🚀 Overview

This project presents a sophisticated, unified AI chatbot application built with Python, leveraging:

- **IBM Watsonx.ai** for large language models (LLMs) and embeddings  
- **LangChain** for robust AI application development  
- **Gradio** for an intuitive web-based user interface  

Designed for versatility, the chatbot offers two primary modes of interaction:

- **Direct Chatbot**: Engage in open-ended conversations powered by a state-of-the-art LLM.  
- **PDF RAG Chatbot**: Upload a PDF and query its contents directly. This enables accurate, context-aware answers grounded in your documents via **Retrieval-Augmented Generation (RAG)**.

This solution is ideal for building intelligent agents capable of both general conversation and precise, document-based responses.

---

## ✨ Features

- **Unified Interface**: Single Gradio app with tabbed navigation between chat modes  
- **Direct LLM Interaction**: Uses `meta-llama/llama-3-70b-instruct` on IBM Watsonx.ai  
- **RAG Pipeline**:
  - PDF document ingestion
  - Recursive character chunking with `RecursiveCharacterTextSplitter`
  - Embedding generation using `ibm/slate-125m-english-rtrvr`
  - Chroma vector store for efficient retrieval
  - Accurate, context-based answers from uploaded documents  
- **Scalable Backend**: Runs on IBM Watsonx.ai for enterprise-grade performance  
- **User-Friendly UI**: Gradio-powered interface accessible to all users  

---

## 🛠️ How It Works

### 🔤 Language Model (LLM) & Embeddings

- `get_llm()` initializes `WatsonxLLM` using the `meta-llama/llama-3-70b-instruct` model  
- `watsonx_embedding()` sets up `WatsonxEmbeddings` with the `ibm/slate-125m-english-rtrvr` model  

### 📄 RAG Pipeline

1. PDF uploaded via Gradio UI  
2. `PyPDFLoader` extracts document text  
3. `RecursiveCharacterTextSplitter` breaks text into chunks  
4. Chunks are embedded using Watsonx.ai embedding model  
5. Chroma DB stores and indexes embeddings  
6. Query retrieves relevant chunks for context  
7. LLM generates a document-grounded response  

### 🖥️ Gradio Interface

- `unified_interface()` creates two tabs:
  - **Direct Chatbot Tab**
  - **PDF RAG Chatbot Tab**
- Wrapped into a single `TabbedInterface` for easy navigation  

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.9+
- `pip` (Python package manager)

### 📦 Installation

```bash
# Clone the repository
git clone <repository_url>
cd <repository_name>

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
