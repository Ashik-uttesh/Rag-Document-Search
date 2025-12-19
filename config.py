# %%
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

# Model Configuration - Using LOCAL models (no API needed!)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free embedding model (local)

# LOCAL LLM - Choose one:
LLM_MODEL = "google/flan-t5-base"  # RECOMMENDED: Small, fast, good quality
# LLM_MODEL = "facebook/opt-1.3b"  # Alternative: Medium size
# LLM_MODEL = "microsoft/phi-2"  # Alternative: High quality but larger

# Pipeline settings
USE_LOCAL_MODEL = True  # Always True for local pipeline
MAX_NEW_TOKENS = 500  # Maximum tokens for answer (increase for longer answers)
MIN_LENGTH = 50  # Maximum length of generated answer
TEMPERATURE = 0.7  # Creativity (0.1 = focused, 1.0 = creative)

# ChromaDB Configuration
CHROMA_DB_DIR = "./chroma_db"  # Where to store the vector database
COLLECTION_NAME = "document_collection"  # Name of the collection

# Document Processing Configuration
CHUNK_SIZE = 1000  # Size of text chunks (in characters)
CHUNK_OVERLAP = 200  # Overlap between chunks

# Retrieval Configuration
TOP_K_RESULTS = 3  # Number of relevant chunks to retrieve

# Streamlit Configuration
PAGE_TITLE = "RAG Document Search"
PAGE_ICON = "📚"

# Hugging Face API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/"

# %%



