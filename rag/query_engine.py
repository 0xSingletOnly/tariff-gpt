# rag/query_engine.py
from typing import Dict, List, Any, Optional
import logging
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from .retriever import AdvancedRAGRetriever
from models.llm import MistralLLM

logger = logging.getLogger(__name__)

class TariffQueryEngine:
    """Engine to perform RAG queries about Trump tariffs impact on Singapore."""
    
    def __init__(
        self, 
        retriever: AdvancedRAGRetriever
    ):
        self.retriever = retriever
        # Initialize Mistral LLM
        self.llm = MistralLLM()
        
    def _format_sources_for_prompt(self, sources: List[Dict[str, Any]]) -> str:
        """Format retrieved sources into a string for the prompt."""
        formatted_sources = []
        
        for i, source in enumerate(sources, 1):
            formatted_source = f"SOURCE {i}:\nTitle: {source['title']}\nSource: {source['source']}\nDate: {source['date']}\n\nContent:\n{source['content']}\n\n"
            formatted_sources.append(formatted_source)
        
        return "\n".join(formatted_sources)
    
    def generate_formatted_response(
        self, 
        query: str,
        use_query_rewriting: bool = True,
        use_compression: bool = False
    ) -> Dict[str, Any]:
        """Generate a response with source attribution using Mistral API."""
        logger.info(f"Processing query: {query}")
        
        # Get sources based on retrieval method
        if use_compression:
            # Get compressed documents for focused information
            docs = self.retriever.retrieve_with_compression(query)
            sources = []
            for doc in docs:
                sources.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "title": doc.metadata.get("title", "Untitled"),
                    "date": doc.metadata.get("date", "Unknown date"),
                    "metadata": doc.metadata
                })
        else:
            # Get sources with full attribution
            sources = self.retriever.retrieve_with_sources(query, rewrite_query=use_query_rewriting)
        
        # Create the sources text for the prompt
        sources_text = self._format_sources_for_prompt(sources)
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["query", "sources"],
            template="""You are an expert economic analyst specializing in international trade and tariff impacts. You've been asked to analyze how Trump's tariff policies might affect Singapore's economy.

Using ONLY the information provided in the sources below, provide a detailed analysis in response to this query:

QUERY: {query}

SOURCES:
{sources}

In your response:
1. Directly address the query using specific information from the sources
2. Include relevant facts, figures, and data points when available
3. Consider impacts on specific Singapore industries mentioned
4. Acknowledge where sources might present different perspectives
5. Cite source numbers when making specific claims
6. Don't include information not present in the sources
7. If the sources don't contain enough information to fully answer the query, acknowledge this limitation

ANALYSIS:"""
        )
        
        # Generate the response using Mistral API
        chain = prompt_template | self.llm
        response = chain.invoke({
            "query": query,
            "sources": sources_text
        })
        
        # Prepare source attribution for the response
        source_attribution = []
        for i, source in enumerate(sources, 1):
            attribution = {
                "id": i,
                "title": source["title"],
                "source": source["source"],
                "date": source["date"],
                "content": source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"]
            }
            source_attribution.append(attribution)
        
        # Return structured response
        result = {
            "query": query,
            "response": response.strip(),
            "sources": source_attribution,
            "rewritten_query": self.retriever.rewrite_query(query) if use_query_rewriting else None
        }
        
        return result