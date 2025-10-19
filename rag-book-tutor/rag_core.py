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
from langchain.chains import ConversationalRetrievalChain

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

@st.cache_data(hash_funcs={list:lambda x:str(x), dict:lambda x:str(x)})
def split_text(_documents):
    # Split pages into smaller chunks (to improve retrieval accuracy)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    print("Create chunkings...")
    for i, doc in enumerate(_documents):
        print(f"Page {i+1} content preview:", doc.page_content[:200])
    chunks = text_splitter.split_documents(_documents)  # list of Documents (each ≤ ~1000 chars)
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

@st.cache_resource
def get_pipe():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.5,
        repetition_penalty=1.1,
        device=-1
    )

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
    # ⚙ Load lightweight model (no Hugging Face key)
    pipe = get_pipe() 
    llm = HuggingFacePipeline(pipeline=pipe)
    # 🧩 Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}  # prompt replaced here
    )

    return qa_chain


