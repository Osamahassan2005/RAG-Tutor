import tempfile
import random
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.chains import RetrievalQA

@st.cache_data(show_spinner=False)
def process_pdf(uploaded_file):
    all_docs=[]
    for file in uploaded_file:
        # Save uploaded PDF to a temp file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        # Load the PDF using LangChain’s PyPDFLoader
        print("Document loading...")
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()# list of Documents, one per page
        all_docs.extend(documents)
    return all_docs

@st.cache_data(show_spinner=False)
def split_text(documents):
    # Split pages into smaller chunks (to improve retrieval accuracy)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    print("Create chunkings...")
    chunks = text_splitter.split_documents(documents)  # list of Documents (each ≤ ~1000 chars)
    return chunks

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
@st.cache_data(show_spinner=False)
def create_embeddings(_chunks):
    # Create embeddings and vector store index
    print('creating embeddings...')
    vector_db = FAISS.from_documents(
    documents=_chunks,
    embedding=embeddings,
    )
    print('embeddings created successfully!')
    retriever = vector_db.as_retriever(search_kwargs={"k":3})
    return retriever


def create_qa_chain(retriever):
    # 🧠 Custom prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful and knowledgeable tutor who answers questions from a textbook.\n\n"
            "Use ONLY the context to answer. If not found, say '❌ Insufficient evidence.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Detailed Answer:"
        )
    )
    # ⚙ Load model locally (no Hugging Face key)
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",   # lightweight model
        tokenizer="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.5,
        repetition_penalty=1.1
    )

    # Wrap the pipeline for LangChain
    llm = HuggingFacePipeline(pipeline=pipe)
    # 🧩 Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",           # simpler and compatible
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def generate_summary(chunks, max_new_tokens=300):
    """
    Generate a summary of the uploaded textbook or chapter.
    Handles long text safely by truncating or chunking input to avoid token overflow.
    """
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        truncation=True,  # ensure safe truncation
        max_length=512,  # model input limit
        min_length=80,
        max_new_tokens=max_new_tokens
    )
    # Combine only limited text from chunks safely
    text_data = " ".join([chunk.page_content for chunk in chunks[:5]])
    # Split text into smaller 400-word pieces
    words = text_data.split()
    chunk_size = 400
    text_parts = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    # Summarize each part and combine results
    summaries = []
    for part in text_parts:
        try:
            summary_piece = summarizer(
                part,
                do_sample=False,
                clean_up_tokenization_spaces=True
            )[0]['summary_text']
            summaries.append(summary_piece)
        except Exception as e:
            summaries.append(f"[⚠ Skipped part due to error: {e}]")
    # Merge all partial summaries into one final summary
    final_summary = " ".join(summaries)
    return final_summary.strip()






