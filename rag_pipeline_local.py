# %%
"""
RAG Pipeline Module - LOCAL VERSION
Uses local Hugging Face models - NO API NEEDED!
Works 100% offline after first download
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Tuple
import torch

class RAGPipelineLocal:
    """RAG pipeline using local models - guaranteed to work!"""
    
    def __init__(self, vector_store, llm_model: str, top_k: int = 3):
        """
        Initialize RAG pipeline with LOCAL model
        
        Args:
            vector_store: VectorStore instance
            llm_model: Hugging Face model name (will download first time)
            top_k: Number of chunks to retrieve
        """
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.top_k = top_k
        
        print(f"\n🔄 Loading LOCAL model: {llm_model}")
        print("⏳ First time will download model (5-8 mins depending on model)")
        print("⚡ After first time, loading is instant!")
        print("-" * 70)
        
        # Check if GPU available
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if self.device == 0 else "CPU"
        print(f"🖥️  Using: {device_name}")
        
        # Load tokenizer and model
        try:
            # For T5-based models (recommended)
            if "t5" in llm_model.lower():
                print("Loading T5 model...")
                self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
                self.generator = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device
                )
            # For other generative models
            else:
                print("Loading generative model...")
                self.generator = pipeline(
                    "text-generation",
                    model=llm_model,
                    device=self.device
                )
            
            print(f"✅ Model loaded successfully!")
            print(f"📦 Model cached at: C:\\Users\\Admin\\.cache\\huggingface\\")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        print("-" * 70)
        print(f"✅ RAG Pipeline (LOCAL) ready!")
        print(f"📊 Retrieving top {top_k} chunks per query\n")
    
    def retrieve_relevant_chunks(self, query: str) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve relevant document chunks for a query
        
        Args:
            query: User question
            
        Returns:
            Tuple of (texts, metadatas)
        """
        print(f"🔍 Retrieving relevant chunks for: '{query}'")
        
        # Search vector store
        results = self.vector_store.search(query, top_k=self.top_k)
        
        # Extract texts and metadata
        texts = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        print(f"✅ Retrieved {len(texts)} relevant chunks\n")
        
        return texts, metadatas
    
    def create_prompt(self, query: str, context_chunks: List[str]) -> str:
        """
        Create an optimized prompt for local models that encourages detailed answers
        
        Args:
            query: User question
            context_chunks: Retrieved relevant text chunks
            
        Returns:
            Formatted prompt
        """
        # Combine context (use more context for better answers)
        context = "\n\n".join(context_chunks[:3])
        
        # Optimize prompt for T5 models with instruction for detailed answers
        if "t5" in self.llm_model.lower():
            prompt = f"""Based on the following context, provide a detailed and complete answer to the question. Include all relevant information from the context.

Context: {context[:2000]}

Question: {query}

Provide a comprehensive answer with complete sentences:"""
        else:
            prompt = f"""Context: {context[:2000]}

Question: {query}

Detailed answer:"""
        
        return prompt
    
    def generate_answer(self, prompt: str, max_tokens: int = 800) -> str:
        """
        Generate answer using LOCAL model
        
        Args:
            prompt: Complete prompt with context and question
            max_tokens: Maximum length of generated answer (in tokens)
            
        Returns:
            Generated answer
        """
        print(f"🤖 Generating answer using LOCAL model (max {max_tokens} tokens)...")
        
        try:
            # Generate with parameters optimized for T5
            if "t5" in self.llm_model.lower():
                result = self.generator(
                    prompt,
                    max_new_tokens=max_tokens,
                    min_new_tokens=80,  # Increased minimum
                    num_beams=5,  # More beams for better quality
                    no_repeat_ngram_size=3,
                    length_penalty=1.2,  # Encourage longer outputs
                    early_stopping=False  # Don't stop early
                )
            else:
                result = self.generator(
                    prompt,
                    max_new_tokens=max_tokens,
                    min_new_tokens=30,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    no_repeat_ngram_size=3
                )
            
            # Extract answer
            if isinstance(result, list) and len(result) > 0:
                answer = result[0]['generated_text'].strip()
                
                # Clean up answer (remove prompt if included)
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
            else:
                answer = "Could not generate answer."
            
            print(f"✅ Answer generated!\n")
            return answer
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"Error: {str(e)}"
    
    def query(self, question: str) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate answer
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        print("\n" + "="*70)
        print(f"❓ Question: {question}")
        print("="*70 + "\n")
        
        # Step 1: Retrieve relevant chunks
        context_chunks, metadatas = self.retrieve_relevant_chunks(question)
        
        if not context_chunks:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": [],
                "context_chunks": []
            }
        
        # Step 2: Create prompt
        prompt = self.create_prompt(question, context_chunks)
        
        # Step 3: Generate answer using LOCAL model (with custom token length)
        answer = self.generate_answer(prompt, max_tokens=800)
        
        # Prepare response
        response = {
            "answer": answer,
            "sources": list(set([meta.get('source', 'Unknown') for meta in metadatas])),
            "context_chunks": context_chunks,
            "metadatas": metadatas
        }
        
        print("="*70)
        print(f"✅ ANSWER: {answer}")
        print(f"\n📚 Sources: {', '.join(response['sources'])}")
        print("="*70 + "\n")
        
        return response

# %%



