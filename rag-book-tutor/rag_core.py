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

@st.cache_data(hash_funcs={list:str, dict:str})
def split_text(_documents):
    # Split pages into smaller chunks (to improve retrieval accuracy)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    print("Create chunkings...")
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
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
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

@st.cache_resource
def get_summarizer():
    # Use a small and fast model (T5 small)
    return pipeline(
        "summarization",
        model="t5-small",
        tokenizer="t5-small",
        framework="pt",
        device=-1  # CPU
    )

def generate_summary(chunks, max_new_tokens=150):
    if not chunks:
        return "❌ No text found to summarize."

    summarizer = get_summarizer()

    # Combine only a limited number of chunks
    text_data = " ".join([chunk.page_content for chunk in chunks[:3]])
    words = text_data.split()
    chunk_size = 300
    text_parts = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    summaries = []
    progress = st.progress(0)

    for i, part in enumerate(text_parts):
        st.write(f"🔹 Summarizing part {i+1}/{len(text_parts)}...")  # Debug print
        try:
            output = summarizer(
                part,
                max_length=max_new_tokens,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            summaries.append(output.strip())
        except Exception as e:
            summaries.append(f"[⚠ Error on part {i+1}: {e}]")

        progress.progress((i + 1) / len(text_parts))

    final_summary = " ".join(summaries)
    return final_summary.strip()

