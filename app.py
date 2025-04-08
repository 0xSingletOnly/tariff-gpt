# app.py
import streamlit as st
import os
from rag.document_processor import DocumentProcessor
from rag.retriever import AdvancedRAGRetriever
from rag.query_engine import TariffQueryEngine

# Set page configuration
st.set_page_config(
    page_title="Trump Tariff Impact Analyzer - Singapore",
    page_icon="🇸🇬",
    layout="wide"
)

# Initialize session state for storing the query engine
if 'query_engine' not in st.session_state:
    with st.spinner("Initializing the Tariff Impact Analysis System..."):
        # Process documents
        processor = DocumentProcessor(chunk_size=8000, chunk_overlap=1600)
        documents = processor.process_all_documents()
        
        # Initialize retriever
        retriever = AdvancedRAGRetriever(
            documents=documents,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            persist_directory="db/chroma",
            k=5  # Retrieve 5 documents
        )
        
        # Initialize query engine
        st.session_state.query_engine = TariffQueryEngine(retriever=retriever)
    st.success("System initialized successfully!")

# Title and introduction
st.title("🇸🇬 Trump Tariff Impact Analysis: Singapore Focus")
st.markdown("""
This tool analyzes how Trump's proposed tariff policies might impact Singapore's economy.
Enter your questions about specific industries, trade impacts, or economic effects.
""")

# Query input
query = st.text_input("Enter your question about Trump's tariffs and Singapore:", 
                     "How would Trump's proposed 60% tariff on Chinese imports affect Singapore's electronics manufacturing sector?")

# Options
with st.expander("Advanced Options"):
    col1, col2 = st.columns(2)
    with col1:
        use_query_rewriting = st.checkbox("Use Query Rewriting", value=True, 
                                        help="Expand queries to include relevant economic terms and concepts")
    with col2:
        use_compression = st.checkbox("Use Context Compression", value=False,
                                     help="Extract only the most relevant parts of retrieved documents")

# Process the query
if st.button("Analyze"):
    if not query:
        st.warning("Please enter a question to analyze.")
    else:
        with st.spinner("Analyzing tariff impacts..."):
            # Generate response
            result = st.session_state.query_engine.generate_formatted_response(
                query=query,
                use_query_rewriting=use_query_rewriting,
                use_compression=use_compression
            )
            
            # Display results
            st.markdown("### Analysis")
            st.markdown(result["response"])
            
            # Show query rewriting if enabled
            if use_query_rewriting and result["rewritten_query"]:
                st.markdown("### Query Understanding")
                st.markdown("I interpreted your question as:")
                st.info(result["rewritten_query"])
            
            # Show sources
            st.markdown("### Sources")
            for source in result["sources"]:
                with st.expander(f"{source['id']}. {source['title']} ({source['source']}, {source['date']})"):
                    st.markdown("**Content Preview:**")
                    st.text(source["content"])

# Add information about the project
st.markdown("---")
st.markdown("### About This Project")
st.markdown("""
This demonstration showcases advanced RAG (Retrieval Augmented Generation) techniques using Mistral AI's ministral-3B model. Features:

- **Hybrid Retrieval**: Combines semantic search and keyword matching
- **Query Rewriting**: Expands economic queries with relevant terminology
- **Source Attribution**: Transparent citation of information sources
- **Singapore Focus**: Specifically analyzes impacts on Singapore's economy
""")