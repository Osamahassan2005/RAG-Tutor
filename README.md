# 📚 RAG Book Tutor

A **Retrieval-Augmented Generation (RAG)** based tutor for textbooks. Users can upload PDFs and ask questions. The system retrieves relevant content and generates answers using a **lightweight local LLM (`google/flan-t5-base`)**, without API keys.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [RAG Core (`rag_core.py`) Functions](#rag-core-rag_corepy-functions)  
6. [File Structure](#file-structure)    
8. [Contributing](#contributing)
   
## Overview

RAG Book Tutor combines **PDF processing, text chunking, embeddings, and QA chain** to answer questions directly from textbooks.

**Workflow:**

1. Upload a PDF file.  
2. Split the PDF pages into smaller chunks for better retrieval.  
3. Convert chunks into vector embeddings using **FAISS**.  
4. Use a retriever to fetch relevant chunks for the user’s question.  
5. Generate answers using a lightweight local LLM.  
6. Optionally, show the source pages for reference.

---

## Features

- Upload PDFs and query them directly.  
- Answers generated **only from the uploaded document**.  
- Lightweight, **local LLM**; no API key required.  
- Conversational QA with **chat history**.  
- Shows **source pages** for verification.  
- Caching for **chunks and embeddings** speeds up repeated queries.  
- Handles large PDFs by splitting into **manageable chunks**.

---

## Installation

1. Install dependencies:

pip install -r requirements.txt

2. Run the app:

streamlit run RAG-tutor/app.py

> Note: Python 3.10+ is required. Streamlit Cloud manages dependencies automatically.

## Usage

1. Open the Streamlit app in your browser.

2. Upload a PDF textbook.

3. Type a question in the input box.

4. Click Submit.

5. The app retrieves relevant sections and shows:

Answer

Sourcepage numbers



## RAG Core (rag_core.py) Functions

Function	Description

1. process_pdf(file_path)	
Reads PDF and converts pages into Document objects with metadata.

2. split_text(_documents) 
Splits document pages into smaller chunks (~500 chars) for better retrieval. Uses @st.cache_data for caching.

3. create_embeddings(chunks)
Converts chunks into vector embeddings using sentence-transformers.

4. create_qa_chain(retriever)	
Creates the QA chain using google/flan-t5-base and a custom prompt. Supports chat_history.

5. generate_summary(text)
Summarizes a section or PDF content. Optional for quick overviews.


## Notes:

Retriever uses FAISS for similarity search.

PromptTemplate ensures answers are strictly based on the document.

Caching improves performance for repeated queries.



## File Structure

rag-book-tutor/
│
├─ app.py                  # Main Streamlit app
├─ rag_core.py             # Core RAG functions
├─ requirements.txt        # Python dependencies
├─ assets/                 # Images, logos, or static files
├─ README.md
└─ .gitignore

## Contributing

1. Fork the repository.


2. Create a new branch (feature-branch).


3. Make your changes and test locally.


4. Submit a pull request with a clear description.


## References

Streamlit Documentation

LangChain Documentation

Hugging Face Transformers

FAISS for Similarity Search

