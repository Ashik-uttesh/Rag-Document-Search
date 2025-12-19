# %%
"""
Vector Store Module
Handles ChromaDB operations and embeddings using Hugging Face
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os

class VectorStore:
    """Manage vector database operations with ChromaDB"""
    
    def __init__(self, db_dir: str, collection_name: str, embedding_model: str):
        """
        Initialize ChromaDB and embedding model
        
        Args:
            db_dir: Directory to store ChromaDB data
            collection_name: Name of the collection
            embedding_model: Hugging Face model for embeddings
        """
        self.db_dir = db_dir
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(db_dir, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_dir)
        
        # Load embedding model (runs locally, FREE!)
        print(f"🔄 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✅ Embedding model loaded successfully!")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document chunks for RAG"}
        )
        
        print(f"✅ ChromaDB initialized at: {db_dir}")
        print(f"📦 Collection: {collection_name}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print(f"✅ Embeddings generated!")
        return embeddings.tolist()
    
    def add_documents(self, chunks: List[Dict[str, any]], source: str = "unknown"):
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of chunk dictionaries from DocumentProcessor
            source: Source document name
        """
        if not chunks:
            print("⚠️ No chunks to add")
            return
        
        print(f"\n🔄 Adding {len(chunks)} chunks to vector store...")
        print("-" * 50)
        
        # Prepare data
        texts = [chunk["text"] for chunk in chunks]
        ids = [f"{source}_chunk_{chunk['id']}" for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Prepare metadata
        metadatas = [
            {
                "source": source,
                "chunk_id": chunk["id"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "length": chunk["length"]
            }
            for chunk in chunks
        ]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"✅ Successfully added {len(chunks)} chunks to vector store")
        print(f"📊 Total documents in collection: {self.collection.count()}")
        print("-" * 50 + "\n")
    
    def search(self, query: str, top_k: int = 3) -> Dict:
        """
        Search for relevant documents using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Dictionary with results
        """
        print(f"\n🔍 Searching for: '{query}'")
        print("-" * 50)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        print(f"✅ Found {len(results['documents'][0])} relevant chunks")
        print("-" * 50 + "\n")
        
        return results
    
    def get_collection_stats(self):
        """Get statistics about the collection"""
        count = self.collection.count()
        print(f"\n📊 Collection Statistics:")
        print("-" * 50)
        print(f"Collection Name: {self.collection_name}")
        print(f"Total Documents: {count}")
        print(f"Storage Location: {self.db_dir}")
        print("-" * 50 + "\n")
        return count
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document chunks for RAG"}
        )
        print(f"✅ Collection '{self.collection_name}' cleared")


# Example usage and testing
if __name__ == "__main__":
    import config
    from document_processor import DocumentProcessor
    
    # Initialize vector store
    vector_store = VectorStore(
        db_dir=config.CHROMA_DB_DIR,
        collection_name=config.COLLECTION_NAME,
        embedding_model=config.EMBEDDING_MODEL
    )
    
    # Test: Process a document and add to vector store
    pdf_path = r"C:\Users\Admin\Desktop\Ashik\NOTES.pdf"  # Update with your PDF
    
    if os.path.exists(pdf_path):
        # Process document
        processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = processor.process_document(pdf_path)
        
        # Add to vector store
        vector_store.add_documents(chunks, source=os.path.basename(pdf_path))
        
        # Get stats
        vector_store.get_collection_stats()
        
        # Test search
        test_query = "What is this document about?"
        results = vector_store.search(test_query, top_k=3)
        
        # Display results
        print("\n📋 Search Results:")
        print("-" * 50)
        for i, doc in enumerate(results['documents'][0]):
            print(f"\n Result {i+1}:")
            print(doc[:200] + "...")
    else:
        print(f"⚠️ PDF file not found: {pdf_path}")
        print("Please update the pdf_path variable")


