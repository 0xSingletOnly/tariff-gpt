# rag/document_processor.py
import os
import re
import yaml
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process raw documents from data/raw into chunks with metadata."""
    
    def __init__(
        self,
        raw_docs_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.raw_docs_dir = raw_docs_dir
        self.processed_dir = processed_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Make sure processed directory exists
        os.makedirs(processed_dir, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ".", " ", ""],
            length_function=len
        )
    
    def load_document(self, file_path: str) -> Tuple[Dict[str, Any], str]:
        """Load a document and separate its YAML metadata and content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split metadata and content
        if content.startswith('---'):
            # Find the end of the YAML front matter
            _, yaml_text, document_text = content.split('---', 2)
            try:
                metadata = yaml.safe_load(yaml_text.strip())
                return metadata, document_text.strip()
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML in {file_path}: {e}")
                return {}, content
        else:
            return {}, content
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess the document text."""
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up financial-specific notation
        # Convert percentage ranges
        text = re.sub(r'(\d+)-(\d+)%', r'\1% to \2%', text)
        
        # Expand financial abbreviations
        financial_terms = {
            r'\bGDP\b': 'Gross Domestic Product (GDP)',
            r'\bFDI\b': 'Foreign Direct Investment (FDI)',
            r'\bWTO\b': 'World Trade Organization (WTO)',
            r'\bIMF\b': 'International Monetary Fund (IMF)',
            r'\bSME\b': 'Small and Medium Enterprise (SME)'
        }
        
        for abbr, full in financial_terms.items():
            # Only replace the first occurrence
            text = re.sub(abbr, full, text, count=1)
        
        return text.strip()
    
    def enhance_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance metadata with additional useful fields."""
        enhanced = metadata.copy()
        
        # Add recency score based on date
        if 'date' in metadata:
            try:
                pub_date = datetime.strptime(metadata['date'], '%Y-%m-%d')
                days_ago = (datetime.now() - pub_date).days
                # Recency score: 1.0 (very recent) to 0.1 (old)
                enhanced['recency_score'] = max(0.1, min(1.0, 1.0 - (days_ago / 365)))
            except (ValueError, TypeError):
                enhanced['recency_score'] = 0.5  # Default if date parsing fails
        
        # Estimate source credibility if not provided
        if 'credibility' not in enhanced:
            # Map common sources to credibility scores
            credibility_map = {
                'bloomberg': 0.9,
                'reuters': 0.9,
                'financial times': 0.9,
                'wall street journal': 0.85,
                'straits times': 0.8,
                'channel news asia': 0.8,
                'south china morning post': 0.75
            }
            
            source = enhanced.get('source', '').lower()
            for key, score in credibility_map.items():
                if key in source:
                    enhanced['credibility'] = score
                    break
            else:
                enhanced['credibility'] = 0.7  # Default credibility
        
        # Create a combined relevance score
        relevance = float(enhanced.get('relevance', 5)) / 10
        credibility = float(enhanced.get('credibility', 0.7))
        recency = enhanced.get('recency_score', 0.5)
        
        # Combined score weighted toward relevance
        enhanced['combined_score'] = (0.6 * relevance) + (0.25 * credibility) + (0.15 * recency)
        
        return enhanced
    
    def chunk_document(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """Split document into chunks while preserving metadata."""
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        # Enhance metadata
        enhanced_metadata = self.enhance_metadata(metadata)
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk_text in enumerate(chunks):
            # Add chunk-specific metadata
            chunk_metadata = enhanced_metadata.copy()
            chunk_metadata.update({
                'chunk_id': i,
                'total_chunks': len(chunks),
                'document_id': f"{enhanced_metadata.get('source', 'unknown')}_{enhanced_metadata.get('date', 'unknown')}"
            })
            
            documents.append(Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            ))
        
        return documents
    
    def process_all_documents(self) -> List[Document]:
        """Process all documents in the raw docs directory."""
        all_documents = []
        
        # Get all markdown files
        for filename in os.listdir(self.raw_docs_dir):
            if filename.endswith('.md'):
                file_path = os.path.join(self.raw_docs_dir, filename)
                logger.info(f"Processing {filename}")
                
                try:
                    # Load and parse the document
                    metadata, content = self.load_document(file_path)
                    
                    # Skip empty documents
                    if not content.strip():
                        logger.warning(f"Empty content in {filename}")
                        continue
                    
                    # Add filename to metadata
                    metadata['filename'] = filename
                    
                    # Process the document
                    document_chunks = self.chunk_document(content, metadata)
                    all_documents.extend(document_chunks)
                    
                    logger.info(f"Created {len(document_chunks)} chunks from {filename}")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
        
        logger.info(f"Processed {len(all_documents)} total chunks from all documents")
        return all_documents
    
    def save_processed_documents(self, documents: List[Document]):
        """Save processed documents for inspection or later use."""
        import json
        
        # Convert to serializable format
        serializable_docs = []
        for doc in documents:
            serializable_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Save to JSON
        output_path = os.path.join(self.processed_dir, "processed_chunks.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, indent=2, default=str)
        
        logger.info(f"Saved processed documents to {output_path}")

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    documents = processor.process_all_documents()
    processor.save_processed_documents(documents)
    print(f"Processed {len(documents)} document chunks")