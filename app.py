import streamlit as st
import os
import tempfile
from rag_engine import MultimodalRAG

st.set_page_config(page_title="Multimodal Doc Intelligence", layout="wide")

st.title("📄🖼️ Multimodal Document Intelligence")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, PNG, or JPG", 
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    process_btn = st.button("Process Documents")

# Session State Management
# Maintains conversation history and vector store across reruns
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Processing
if process_btn:
    if not api_key:
        st.error("Please provide an OpenAI API Key.")
    elif not uploaded_files:
        st.error("Please upload at least one document.")
    else:
        with st.spinner("Processing documents... This may take a while for images."):
            # Initialize RAG
            st.session_state.rag = MultimodalRAG(openai_api_key=api_key)
            
            # Save uploaded files to temp
            temp_files = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_files.append(tmp.name)
            
            # Create Vector Store
            try:
                st.session_state.rag.create_vector_store(temp_files)
                st.success("Documents processed successfully!")
            except Exception as e:
                st.error(f"Error processing documents: {e}")
            finally:
                # Cleanup temp files
                for f in temp_files:
                    try:
                        os.remove(f)
                    except:
                        pass

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img_data in message["images"]:
                st.image(f"data:image/jpeg;base64,{img_data}")

if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.rag:
        st.error("Please upload and process documents first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag.query(prompt)
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Check for source images
                    retrieved_images = []
                    if "source_documents" in response:
                        for doc in response["source_documents"]:
                            if doc.metadata.get("type") == "image":
                                img_data = doc.metadata.get("image_data")
                                if img_data:
                                    st.image(f"data:image/jpeg;base64,{img_data}", caption="Retrieved Image Context")
                                    retrieved_images.append(img_data)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "images": retrieved_images
                    })
                except Exception as e:
                    if "insufficient_quota" in str(e):
                        st.error("OpenAI API Quota Exceeded. Please check your billing details at platform.openai.com.")
                    else:
                        st.error(f"An error occurred: {e}")
