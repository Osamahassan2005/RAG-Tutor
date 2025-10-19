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
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.5,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # 🧩 Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}  # prompt replaced here
    )

    return qa_chain

import streamlit as st
from transformers import pipeline
from typing import List

# Cache the summarizer so the model is loaded only once per process
@st.cache_resource
def get_summarizer(model_name: str = "sshleifer/distilbart-cnn-12-6"):
    return pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
        device=-1
    )

def generate_summary(chunks: List, 
                     model_name: str = "sshleifer/distilbart-cnn-12-6",
                     total_word_limit: int = 1500,
                     chunk_word_size: int = 400,
                     max_new_tokens: int = 150):
    """
    Summarize safely:
      - Use cached pipeline
      - Limit total words taken from chunks (total_word_limit)
      - Split into word chunks (chunk_word_size)
      - Summarize each chunk and join results
    Returns the final summary string.
    """
    if not chunks:
        return "❌ No text found to summarize."

    # Build a text budget from chunks (stop when we've reached the word budget)
    words = []
    for doc in chunks:
        # protect against non-text chunks
        text = getattr(doc, "page_content", None)
        if not text:
            continue
        words.extend(text.split())
        if len(words) >= total_word_limit:
            break
    if not words:
        return "❌ No text content available in the provided chunks."

    # Trim to the budget and split into smaller parts
    words = words[:total_word_limit]
    parts = [" ".join(words[i:i+chunk_word_size]) for i in range(0, len(words), chunk_word_size)]

    # Get the cached summarizer (may raise; we catch below)
    try:
        summarizer = get_summarizer(model_name)
    except Exception as e:
        # If model load fails, try a very small fallback
        try:
            summarizer = get_summarizer("t5-small")
            model_name = "t5-small"
        except Exception as e2:
            return f"❌ Failed to load summarizer models: {e} ; {e2}"

    summaries = []
    progress = None
    try:
        progress = st.progress(0)
    except Exception:
        progress = None

    for i, part in enumerate(parts):
        try:
            # request a summary for this part; tune max_length/min_length if necessary
            out = summarizer(
                part,
                max_length=max_new_tokens,
                min_length=30,
                do_sample=False,
                clean_up_tokenization_spaces=True
            )
            # most summarizers return list of dicts with 'summary_text' or 'generated_text'
            summary_piece = out[0].get("summary_text") or out[0].get("generated_text") or str(out[0])
            summaries.append(summary_piece.strip())
        except Exception as e:
            # If a chunk fails, skip it but record an error note
            summaries.append(f"[⚠ skipped part due to: {e}]")

        # update progress
        if progress:
            progress.progress(int((i + 1) / len(parts) * 100))

    # Merge and optionally run a final short summarization pass (optional)
    final_summary = " ".join([s for s in summaries if s])
    return final_summary.strip()



















