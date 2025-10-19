import os
import torch
import shutil 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import streamlit as st
import hashlib
from PIL import Image
#existing imports from rag_core
from rag_core import process_pdf, split_text, create_embeddings, create_qa_chain

cache_paths = [
    os.path.expanduser("~/.cache/torch"),
    os.path.expanduser("~/.cache/huggingface/transformers")
]

def clear_cache():
    st.write("🧹 Clearing cache folders...")
    for path in cache_paths:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                st.write(f"✅ Cleared cache: {path}")
            except Exception as e:
                st.write(f"⚠️ Could not clear {path}: {e}")

    # Clear PyTorch internal cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧠 Torch CUDA cache cleared.")
# ---- Apply custom CSS ----
st.markdown("""
     <style>
      .stApp {
                color: #333; /* dark gray */
                font-family: 'Poppins', sans-serif;
                font-size: 16px;
                text-align: left;
                padding: 0px;
                border-radius: 2px;
                margin: 0px;
                background-color: #fff
            } 
            .stApp header { 
                background-color: #fff;
                color: #6a0dad;
                padding: 10px;
                text-align: left;
                font-size: 24px;
                font-weight: bold; 
            }
            .stApp h1 {
                color:#1a1a1a; /* dark gray */
                font-family: 'Poppins',sans-serif;
                font-size: 46px;
                font-weight: bold;
                text-align: left;
            }
            .stApp h2, .stApp h3, .stApp h4 {
                color: #6a0dad; /* orange */
                font-size: 28px;
                font-weight: bold;
                text-align: left;
            }
            .stApp p {
                color: #333; /* dark gray */
                font-size: 18px;
                line-height: 1.5;
                text-align: left;
            }
            .stApp img {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                margin: 20px 0;
            }
            .stApp button {
                background-color: #6a0dad; /*#1E56A0; orange */
                color: #ffffff !important; /* white */
                border: none;
                border-radius: 7px;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }   
            .stApp button:hover {
                transition: background-color 0.3s ease;
                color: #fff; /* white */
                background-color: #9966cc; /* darker orange */
           }
           .stApp p  {
                background-color: #5b21b6;
                color: #ffffff !important; /* white */
                }
           
            /* Sidebar */
            [data-testid="stSidebar"] {
            background-color: #6a0dad;
            color: #6a0dad ;
            padding: 20px;
            border-radius: 10px 10px;
            margin: 2px;
        }
            section[data-testid="stSidebar"] button {
                background-color: #6a0dad; /* white */
                color: #fff !important; /* orange */
                border: none;
                border-radius: 7px;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            section[data-testid="stSidebar"] p {
                background-color: #6a0dad; /* white */
                color: #fff !important; /* orange */
            }
            section[data-testid="stSidebar"] h2{
                color: ##0A3D91; /* white */
                font-size: 24px;
                font-weight: bold;
                text-align: left;
            }
            
            [data-testid="stSidebar"]
            [data-testid="stRadio"] label {
                color: #fff; /* white */
                font-size: 18px;
                font-weight: bold;
                text-align: left;
                padding: 5px;
            }
            
            input[type="text"],
            textarea {
                background-color: #fff; /* white */
                color: #333; /* dark gray */
                border: 2px solid #6a0dad;
                caret-color: #333; /* dark gray */
                border-radius: 5px;
                padding: 17px;
                font-size: 16px;
                width: 100%;
                outline: none;
                box-sizing: border-box;
                margin-bottom: 20px;
                transition: border-color 0.3s ease;
            }
            input[type="text"]:hover,
            textarea:hover {
                border-color:#6a0dad; /* orange */-----------
                outline: none;
                background-color: #f5f5f5; /* light gray */
            }
            input[type="text"]:focus,
            textarea:focus {
                border-color: #6a0dad; /* orange */
                outline: none;
                background-color: #f5f5f5; /* light gray */
            }
            .stApp .stTextInput label {
                color: #333 ; /* dark gray */
                font-size: 18px;
                font-weight: bold;
                text-align: left;
                padding: 5px;
           }
            
            /* Container styling */
            div[data-baseweb="select"] > div {
                background-color: #fff !important;  /* white bg */
                color: #333 !important;             /* dark text */
                caret-color: #333 !important;    /* dark caret */
                border: 2px solid #6a0dad !important;  /* orange border */
                border-radius: 5px !important;
                padding: 0px 12px !important;
                font-size: 16px !important;
                width: 100% !important;
                margin-bottom: 20px !important;
                transition: border-color 0.3s ease !important;    
              
            /* Hover on container */
            div[data-baseweb="select"] > div:hover {
                border-color: #0A3D91 !important;   /* darker orange */
                background-color: #fafafa !important; /* very light gray */
              }
            

             </style>
            """,
            unsafe_allow_html=True)

#app
st.title("📚 Welcome to RAG Book Tutor")
if st.button("🧹 Reset Streamlit Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("✅ Streamlit cache cleared. Please upload your file again.")
    st.stop()
# PDF upload via Streamlit
uploaded_file = st.sidebar.file_uploader("Upload PDF Textbook",type=["pdf"],accept_multiple_files=True)
mode = st.sidebar.radio("Select Mode", ["Home","Q&A","Clear-Cache"])

# Get the absolute path of the current directory (where app.py is)
def load_image(image_name):
     base_path = os.path.join(os.path.dirname(__file__), "assets")
     image_path = os.path.join(base_path, image_name)
     return Image.open(image_path)

# --- small helper: signature to detect changed uploads ---
def _files_signature(files):
    """
    Returns a stable signature (tuple) for the uploaded files based on filename + sha256 of content.
    This is used to detect when the user uploads new files so we can re-run expensive processing only then."""
    sig = []
    for f in files:
        # ensure we can read and then rewind
        f.seek(0)
        data = f.read()
        h = hashlib.sha256(data).hexdigest()
        f.seek(0)
        sig.append((f.name, h))
    return tuple(sig)
     # Initialize session keys once
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- end helper ---

if uploaded_file is not None:
    # normalize to a list even if a single file is provided
    files = uploaded_file if isinstance(uploaded_file, list) else [uploaded_file]

    # compute signature and compare with session state to avoid reprocessing unchanged uploads
    current_sig = _files_signature(files)
    prev_sig = st.session_state.get("uploaded_sig")
    if prev_sig != current_sig:
        # New upload or changed files -> (re)process and store results in session_state
        with st.spinner("*Processing PDF...*"):
            documents = process_pdf(files)
            st.session_state["documents"] = documents

            with st.spinner("Splitting text..."):
                chunks = split_text(documents)
                st.session_state["chunks"] = chunks

            if chunks:
                with st.spinner("Creating embeddings..."):
                    #st.write('Starting to create embeddings, this may take a few minutes...')
                    retriever = create_embeddings(chunks)
                    st.session_state["retriever"] = retriever
                    #st.write('Embeddings created successfully!')

                with st.spinner("Setting up QA chain..."):
                    qa_chain = create_qa_chain(retriever)
                    st.session_state["qa_chain"] = qa_chain

            else:
                # save empty to avoid KeyError later
                st.session_state["chunks"] = []
                st.session_state["retriever"] = None
                st.session_state["qa_chain"] = None

        # update signature after successful processing
        st.session_state["uploaded_sig"] = current_sig
    else:
        # same files as before — reuse cached objects
        documents = st.session_state.get("documents", [])
        chunks = st.session_state.get("chunks", [])
        retriever = st.session_state.get("retriever")
        qa_chain = st.session_state.get("qa_chain")


    if mode == 'Home':
        st.image(load_image("home.webp"))
        st.markdown("""
    ### How to Use This App:

    1. *Upload Your Documents*  
       - Click on the *Upload* button in the sidebar.  
       - You can upload PDFs, text files, or images.  

    2. *Ask Questions*  
       - Type your question in the *text box* below.  
       - Click *Submit* to get answers from your uploaded documents.  

    3. *View Answers*  
       - The app will retrieve relevant information and provide a clear answer.  
       - You can scroll through multiple answers if available.  

    4. *Clear or Upload New Files*  
       - Use the *Clear* button to remove old files.  
       - Upload new documents anytime to get answers from different sources.  

    5. *Tips for Best Results*  
       - Make your questions *specific* for accurate answers.  
       - Ensure uploaded documents are *clear and readable*.  
       - For large files, processing may take a few seconds.  

    💡 *Note:* This app uses AI-powered retrieval, so answers are based on your uploaded documents.
    """)
    elif mode == "Clear-Cache":
         if st.button("Clear cache"):
            clear_cache()
            st.rerun()

    elif st.session_state.get("chunks"):
        if mode == 'Q&A':
           st.image(load_image("qa.jpg"))
           question = st.text_input("Enter your question about the textbook:")
           if st.button("Get Answer") and question:
               if qa_chain is None:
                   st.error("QA chain is not ready. Please re-upload the PDF or wait until embeddings finish.")
               else:
                    with st.spinner("Retrieving answer..."):
                         result = qa_chain.invoke({
                         "question": question,
                         "chat_history": st.session_state["chat_history"]
                          })

                    # Extract answer safely
                    answer = result.get("answer", result.get("result", result.get("output_text", "No answer found")))

                    st.markdown("### Answer")
                    st.write(answer)

                    # 📚 Show sources if available
                    if result.get("source_documents"):
                       st.markdown("### 📄 Sources")
                       for i, doc in enumerate(result["source_documents"], start=1):
                           page_num = doc.metadata.get("page", doc.metadata.get("page_number", "N/A"))
                           st.markdown(f"**Source {i}:** Page {page_num}")
                    # ✅ Update chat history
                    st.session_state["chat_history"].append((question, answer))
    elif st.session_state.get("documents"):
        st.warning("No chunks were created from the document. Please check the document content.")
























