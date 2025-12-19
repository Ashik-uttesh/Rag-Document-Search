# %%
"""
Document Processor Module
Handles PDF reading, text extraction, and chunking
"""

from pypdf import PdfReader
from typing import List, Dict
import os

class DocumentProcessor:
    """Process PDF documents and split them into chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, pdf_path: str) -> str:
        """
        Load and extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            # Extract text from each page
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            print(f"✅ Successfully loaded PDF: {os.path.basename(pdf_path)}")
            print(f"📄 Total pages: {len(reader.pages)}")
            print(f"📝 Total characters: {len(text)}")
            
            return text
        
        except Exception as e:
            print(f"❌ Error loading PDF: {str(e)}")
            return ""
    
    def split_text_into_chunks(self, text: str) -> List[Dict[str, any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to split
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Get chunk end position
            end = start + self.chunk_size
            
            # Extract chunk
            chunk_text = text[start:end]
            
            # Create chunk with metadata
            chunk = {
                "id": chunk_id,
                "text": chunk_text.strip(),
                "start_char": start,
                "end_char": end,
                "length": len(chunk_text.strip())
            }
            
            chunks.append(chunk)
            
            # Move start position (with overlap)
            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1
        
        print(f"✅ Text split into {len(chunks)} chunks")
        print(f"📊 Average chunk size: {sum(c['length'] for c in chunks) / len(chunks):.0f} characters")
        
        return chunks
    
    def process_document(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Complete processing pipeline: load PDF and split into chunks
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text chunks with metadata
        """
        print(f"\n🔄 Processing document: {os.path.basename(pdf_path)}")
        print("-" * 50)
        
        # Load PDF
        text = self.load_pdf(pdf_path)
        
        if not text:
            print("❌ No text extracted from PDF")
            return []
        
        # Split into chunks
        chunks = self.split_text_into_chunks(text)
        
        print("-" * 50)
        print("✅ Document processing complete!\n")
        
        return chunks


# Example usage and testing
if __name__ == "__main__":
    # Import config
    import config
    
    # Create processor
    processor = DocumentProcessor(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    # Test with a sample PDF (update path to your PDF)
    pdf_path = r"C:\Users\Admin\Desktop\Ashik\NOTES.pdf"  # Change this to your PDF path
    
    if os.path.exists(pdf_path):
        chunks = processor.process_document(pdf_path)
        
        # Display first chunk as example
        if chunks:
            print("\n📋 Example of first chunk:")
            print("-" * 50)
            print(chunks[0]["text"][:300] + "...")
    else:
        print(f"⚠️ PDF file not found: {pdf_path}")
        print("Please update the pdf_path variable with your PDF file path")


