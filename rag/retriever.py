# rag/retriever.py
import os
from typing import List, Dict, Any, Optional
import logging
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor  
from langchain.prompts import PromptTemplate
from models.llm import MistralLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRAGRetriever:
    """Implements advanced RAG retrieval with hybrid search and query transformation."""
    
    def __init__(
        self,
        documents: List[Document],
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "db/chroma",
        k: int = 10
    ):
        self.k = k
        self.documents = documents
        self.persist_directory = persist_directory
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'mps' if os.environ.get('USE_CUDA', 'True').lower() == 'true' else 'cpu'}
        )
        
        # Initialize Mistral LLM for query rewriting and compression
        self.mistral_llm = MistralLLM()
        
        # Set up vector store
        self._setup_vector_store()
        
        # Set up BM25 retriever
        self._setup_bm25_retriever()
        
        # Set up hybrid retriever
        self._setup_hybrid_retriever()
    
    def _setup_vector_store(self):
        """Set up and populate the vector store."""
        # Create vector store directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        logger.info(f"Creating Chroma vector store with {len(self.documents)} documents")
        self.vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vector_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )
        logger.info("Vector store setup complete")
    
    def _setup_bm25_retriever(self):
        """Set up BM25 retriever for lexical search."""
        logger.info("Setting up BM25 retriever")
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = self.k
        logger.info("BM25 retriever setup complete")
    
    def _setup_hybrid_retriever(self):
        """Set up hybrid retriever combining vector and BM25 search."""
        logger.info("Setting up hybrid retriever")
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]  # Weight in favor of semantic search
        )
        logger.info("Hybrid retriever setup complete")
    
    def rewrite_query(self, query: str) -> str:
        """Expand and improve the query for better retrieval using Mistral API."""
        logger.info(f"Rewriting query: {query}")
        
        # Template for query rewriting specific to economic/tariff domain
        prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""You are an expert in international trade and economics. Your job is to rewrite the following query to make it more effective for retrieving relevant information about Trump's tariff policies and their impact on Singapore.

Original query: {query}

Expand this query to include:
1. Relevant economic terms and synonyms
2. Specific industries or sectors that might be affected
3. Related economic concepts
4. Alternative phrasings

Do not add information that isn't implied by the original query. Respond with only the rewritten query, no explanations."""
        )
        
        # Generate the rewritten query using Mistral
        chain = prompt_template | self.mistral_llm
        rewritten_query = chain.invoke({"query": query})
        
        # Clean up the response
        rewritten_query = rewritten_query.strip()
        logger.info(f"Rewritten query: {rewritten_query}")
        
        return rewritten_query
    
    def create_document_compressor(self) -> DocumentCompressorPipeline:
        """Create a document compressor that extracts the most relevant parts of documents."""
        # Template for extraction
        prompt_template = PromptTemplate(
            input_variables=["question", "document"],
            template="""You are an expert assistant that helps extract the most relevant information from documents about Trump's tariff policies and their impact on Singapore.

User question: {question}

Document text: {document}

Extract only the parts of the document that are relevant to answering the question. Include specific facts, figures, and analysis about tariffs and Singapore. Don't exclude important economic data or impact assessments.

RELEVANT INFORMATION:"""
        )
        
        # Create document compressor using Mistral LLM
        return LLMChainExtractor.from_llm(self.mistral_llm, prompt=prompt_template)
    
    def retrieve_with_sources(self, query: str, rewrite_query: bool = True) -> List[Dict[str, Any]]:
        """Retrieve documents with full source attribution and metadata."""
        # Optionally rewrite the query
        if rewrite_query:
            processed_query = self.rewrite_query(query)
        else:
            processed_query = query
        
        # Retrieve documents using hybrid retrieval
        retrieved_docs = self.hybrid_retriever.get_relevant_documents(processed_query)
        
        # Format with sources
        results = []
        for doc in retrieved_docs:
            # Extract and format source information
            metadata = doc.metadata.copy()
            source_info = {
                "content": doc.page_content,
                "source": metadata.get("source", "Unknown"),
                "title": metadata.get("title", "Untitled"),
                "date": metadata.get("date", "Unknown date"),
                "relevance_score": metadata.get("combined_score", 0.5),
                "metadata": metadata
            }
            results.append(source_info)
        
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results

    def retrieve_with_compression(self, query: str) -> List[Document]:
        """Retrieve and compress documents to focus on most relevant parts."""
        # First retrieve with hybrid retrieval
        raw_docs = self.hybrid_retriever.get_relevant_documents(query)
        
        # Create compressor
        compressor = self.create_document_compressor()
        
        # Create compression pipeline
        pipeline = DocumentCompressorPipeline(
            transformers=[compressor]
        )
        
        # Compress documents
        compressed_docs = pipeline.transform_documents(raw_docs, query)
        
        return compressed_docs