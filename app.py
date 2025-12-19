# %%
"""
RAG Document Search - Streamlit Application
A simple web interface for document Q&A using RAG
"""

import streamlit as st
import os
from pathlib import Path

# Import our modules
import config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline_local import RAGPipelineLocal

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag' not in st.session_state:
    st.session_state.rag = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

# Title and description
st.title("📚 RAG Document Search System")
st.markdown("Upload PDF documents and ask questions about them using AI!")

# Sidebar for document management
with st.sidebar:
    st.header("📄 Document Management")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to query"
    )
    
    # Initialize system button
    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Initializing vector store and AI model..."):
            try:
                # Initialize vector store
                st.session_state.vector_store = VectorStore(
                    db_dir=config.CHROMA_DB_DIR,
                    collection_name=config.COLLECTION_NAME,
                    embedding_model=config.EMBEDDING_MODEL
                )
                
                # Initialize RAG pipeline
                st.session_state.rag = RAGPipelineLocal(
                    vector_store=st.session_state.vector_store,
                    llm_model=config.LLM_MODEL,
                    top_k=config.TOP_K_RESULTS
                )
                
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")
    
    # Process documents button
    if uploaded_files and st.session_state.vector_store:
        if st.button("📥 Process Documents", type="secondary"):
            processor = DocumentProcessor(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    # Process document
                    chunks = processor.process_document(temp_path)
                    
                    # Add to vector store
                    st.session_state.vector_store.add_documents(
                        chunks,
                        source=uploaded_file.name
                    )
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                    st.success(f"✅ Processed: {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            st.session_state.documents_loaded = True
            status_text.text("✅ All documents processed!")
    
    # Statistics
    if st.session_state.vector_store:
        st.markdown("---")
        st.subheader("📊 Database Stats")
        count = st.session_state.vector_store.get_collection_stats()
        st.metric("Total Chunks", count)
    
    # Clear database button
    if st.session_state.vector_store:
        st.markdown("---")
        if st.button("🗑️ Clear Database", type="secondary"):
            st.session_state.vector_store.clear_collection()
            st.session_state.documents_loaded = False
            st.session_state.chat_history = []
            st.success("✅ Database cleared!")
            st.rerun()

# Main chat interface
st.markdown("---")
st.header("💬 Ask Questions")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat['question'])
    with st.chat_message("assistant"):
        st.write(chat['answer'])
        with st.expander("📚 View Sources"):
            st.write(f"**Sources:** {', '.join(chat['sources'])}")

# Chat input
if st.session_state.rag and st.session_state.documents_loaded:
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        # Display user question
        with st.chat_message("user"):
            st.write(user_question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag.query(user_question)
                    
                    # Display answer
                    st.write(result['answer'])
                    
                    # Display sources
                    with st.expander("📚 View Sources"):
                        st.write(f"**Sources:** {', '.join(result['sources'])}")
                        st.markdown("**Retrieved Chunks:**")
                        for i, chunk in enumerate(result['context_chunks'][:2]):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text(chunk[:300] + "...")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': result['answer'],
                        'sources': result['sources']
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")

else:
    # Instructions
    st.info("👆 Please initialize the system and upload documents using the sidebar to get started!")
    
    # Show instructions
    with st.expander("📖 How to use"):
        st.markdown("""
        ### Getting Started:
        1. **Initialize System**: Click the "🚀 Initialize System" button in the sidebar
        2. **Upload Documents**: Use the file uploader to select your PDF files
        3. **Process Documents**: Click "📥 Process Documents" to add them to the database
        4. **Ask Questions**: Type your questions in the chat box below
        
        ### Tips:
        - Ask specific questions for better answers
        - You can upload multiple documents at once
        - Use "Explain in detail..." for longer answers
        - Check the sources to see which parts of documents were used
        
        ### Examples:
        - "What is the main topic of this document?"
        - "Explain in detail what a learning automaton is"
        - "Summarize the key points from the document"
        - "What are the important concepts mentioned?"
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit, ChromaDB, and Hugging Face Transformers"
    "</div>",
    unsafe_allow_html=True
)

# %%



