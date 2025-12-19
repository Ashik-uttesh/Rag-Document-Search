# RAG Document Search System

A document Q&A system using Retrieval-Augmented Generation (RAG) with local LLMs.

## Features
- 📄 PDF document processing and chunking
- 🔍 Semantic search using ChromaDB
- 🤖 Local LLM for answer generation (no API needed!)
- 💬 Streamlit web interface
- 📚 Multi-document support

## Tech Stack
- **Vector Database:** ChromaDB
- **Embeddings:** sentence-transformers
- **LLM:** Google FLAN-T5 (local)
- **Frontend:** Streamlit
- **Python:** 3.10

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR-USERNAME/rag-document-search.git
cd rag-document-search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Usage
1. Click "Initialize System" in sidebar
2. Upload PDF documents
3. Click "Process Documents"
4. Ask questions in the chat!

## Project Structure
```
├── app.py                    # Streamlit web interface
├── config.py                 # Configuration settings
├── document_processor.py     # PDF processing and chunking
├── vector_store.py          # ChromaDB operations
├── rag_pipeline_local.py    # RAG pipeline with local LLM
└── requirements.txt         # Python dependencies
```

## Screenshots
<img width="1366" height="599" alt="Screenshot (189)" src="https://github.com/user-attachments/assets/0d7a6331-e99c-40bf-b463-ba4ef33dc14c" />
<img width="300" height="424" alt="Screenshot (190)copy" src="https://github.com/user-attachments/assets/4cd7353f-ec99-4d2c-b13a-7bbcd41af5af" />                             
<img width="299" height="470" alt="Screenshot (191)copy" src="https://github.com/user-attachments/assets/75dd7852-2449-4c02-8e04-4de827074d29" />
<img width="302" height="285" alt="Screenshot (193)copy" src="https://github.com/user-attachments/assets/3de934ff-e634-4038-b5d1-027965afe86b" />
<img width="1366" height="679" alt="Screenshot (194)" src="https://github.com/user-attachments/assets/ce37c3ff-bb4b-4451-8bd1-433fd6b9b4c1" />


<img width="1366" height="768" alt="Screenshot (195)" src="https://github.com/user-attachments/assets/c05cd66c-81f7-4c56-a72c-56b5d782254b" />
<img width="1366" height="768" alt="Screenshot (197)" src="https://github.com/user-attachments/assets/a88a7542-db4a-4e2a-a989-f809ca3b304d" />



## Future Improvements
- [ ] Support for more document formats (DOCX, TXT)
- [ ] Better answer length control
- [ ] Multi-language support
- [ ] Export chat history

## Author
Ashik
