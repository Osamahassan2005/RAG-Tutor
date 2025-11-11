import os
#import torch
import shutil 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import streamlit as st
import hashlib
from PIL import Image
#existing imports from rag_core
from rag_core import process_pdf, split_text, create_embeddings, create_qa_chain

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

# PDF upload via Streamlit
uploaded_file = st.sidebar.file_uploader("Upload PDF Textbook",type=["pdf"],accept_multiple_files=True)
mode = st.sidebar.radio("Select Mode", ["Home","Q&A"])

# Get the absolute path of the current directory (where app.py is)
def load_image(image_name):
     base_path = os.path.join(os.path.dirname(__file__), "assets")
     image_path = os.path.join(base_path, image_name)
     return Image.open(image_path)
# def clear_chat():
#     st.session_state.messages = []
#     st.session_state.conversation = None
#     st.session_state.chat_history = []
#     st.success("💬 Chat history cleared!")
#     st.rerun()

# def clear_cache():
#     st.cache_data.clear()
#     st.cache_resource.clear()
#     st.success("✅ Streamlit cache cleared! Please re-upload your document.")
#     st.rerun()

# def clear_all_caches():
#     cache_paths = [
#         os.path.expanduser("~/.cache/torch"),
#         os.path.expanduser("~/.cache/huggingface/transformers"),
#         os.path.expanduser("~/.streamlit/cache")
#     ]
#     for path in cache_paths:
#         if os.path.exists(path):
#             try:
#                 shutil.rmtree(path)
#             except Exception as e:
#                 print(f"⚠️ Could not clear {path}: {e}")
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     clear_cache()
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
# Remove all clear_cache and clear_all_caches functions
# Keep only clear_chat for conversation history
def clear_chat():
    st.session_state.messages = []
    st.session_state.conversation = None
    st.session_state.chat_history = []
    st.success("💬 Chat history cleared!")
    st.rerun()
# Add this to your sidebar for debugging
with st.sidebar.expander("🔧 System Status"):
    st.write("*Session State:*")
    for key in ["uploaded_sig", "documents", "chunks", "retriever", "qa_chain"]:
        exists = key in st.session_state
        status = "✅" if exists and st.session_state[key] is not None else "❌"
        st.write(f"{status} {key}: {exists}")
    
    if st.button("🔄 Force Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("💥 Complete reset done!")
        st.rerun()
# Enhanced file processing with automatic state cleanup
if uploaded_file is not None:
    files = uploaded_file if isinstance(uploaded_file, list) else [uploaded_file]
    current_sig = _files_signature(files)
    prev_sig = st.session_state.get("uploaded_sig")
    
    if prev_sig != current_sig:
        # NEW FILE DETECTED - COMPLETELY RESET STATE
        #st.info("🔄 New file detected - resetting processing state...")
        
        # Comprehensive state cleanup
        processing_keys = ["uploaded_sig", "documents", "chunks", "retriever", "qa_chain"]
        for key in processing_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear any existing cache
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Now process the new file
        with st.spinner("📄 Processing PDF document..."):
            documents = process_pdf(files)
            st.session_state["documents"] = documents

            with st.spinner("✂ Splitting text into chunks..."):
                chunks = split_text(documents)
                st.session_state["chunks"] = chunks

            if chunks:
                with st.spinner("🔧 Creating embeddings..."):
                    retriever = create_embeddings(chunks)
                    st.session_state["retriever"] = retriever

                with st.spinner("⚙ Setting up QA system..."):
                    qa_chain = create_qa_chain(retriever)
                    st.session_state["qa_chain"] = qa_chain
                    
                #st.success(f"✅ Document processed successfully! Created {len(chunks)} chunks.")
                st.session_state["uploaded_sig"] = current_sig
                
            else:
                #st.error("❌ No text chunks could be created from the document.")
                # Don't set uploaded_sig so it will retry next time
                st.session_state["chunks"] = []
                st.session_state["retriever"] = None  
                st.session_state["qa_chain"] = None

             
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
    elif st.session_state.get("chunks"):
        # Fix the QA chain invocation and answer extraction
        if mode == 'Q&A':
            st.header("📝 Q&A")
    
            # State validation
            if "uploaded_sig" not in st.session_state:
                st.info("📁 Please upload a PDF document to begin")
            elif "qa_chain" not in st.session_state or st.session_state["qa_chain"] is None:
                st.warning("🔄 Document processing incomplete. Please wait or re-upload.")
            else:
                st.success("✅ Document ready for questions!")
            
            question = st.text_input("Enter your question about the textbook:")
            
            if st.button("Get Answer") and question:
                # Validate state before processing
                if "qa_chain" not in st.session_state or st.session_state["qa_chain"] is None:
                    st.error("❌ System not ready. Please upload a document first.")
                else:
                    with st.spinner("🔍 Searching for answers..."):
                        try:
                            qa_chain = st.session_state["qa_chain"]
                            
                            # Convert chat history to correct format
                            chat_history = []
                            if "chat_history" in st.session_state:
                                for entry in st.session_state["chat_history"]:
                                    if isinstance(entry, dict):
                                        chat_history.append((entry["question"], entry["answer"]))
                                    elif isinstance(entry, tuple) and len(entry) == 2:
                                        chat_history.append(entry)
                    
                            result = qa_chain.invoke({
                                "question": question,
                                "chat_history": chat_history
                            })
                    
                            # Extract and display answer
                            answer = result.get("answer", "No answer found.")
                            st.markdown("### 💡 Answer")
                            st.write(answer)
                    
                            # Update chat history as tuples
                            if "chat_history" not in st.session_state:
                                st.session_state["chat_history"] = []
                            st.session_state["chat_history"].append((question, answer))
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.info("💡 Try clearing cache and re-uploading the document.")

    elif st.session_state.get("documents"):
        st.warning("No chunks were created from the document. Please check the document content.")


