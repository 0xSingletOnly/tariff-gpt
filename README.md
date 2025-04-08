# Tariff GPT
It's April 2025, and DJT has announced an extensive tariffs plan on many economies. As a Singaporean, I'm similarly shocked that we are hit with a 10% tariff even despite of the healthy relationship between these two countries.

Tariff-GPT is my attempt to use AI to understand these tariffs and their impact on Singapore. I will be detailed my steps in data collection, data processing, RAG and agentic implementation. 

## Data collection
Given the small dataset that I wanted to collect, I manually googled news articles covering Trump's tariffs. Here is the breakdown of article types:
- 4 articles on general Trump tariff policies
- 5 articles specifically analyzing Singapore impact
- 4 articles on broader Asia/ASEAN implications
- 2-3 expert opinion pieces/editorials

I collected these articles from a variety of sources, including CNA and The Straits Times for Singapore coverage, and Bloomberg/Financial Times for global coverage.

## Data processing
After collecting the data, we want to process them for better performance in RAG later on. We will do the following:
- Implement document chunking with strategic overlap: Chunk each article into smaller "pieces" so that our model is able to better focus. This is less of an issue when we use a bigger model with longer context length, but we are using a 3B model.
- Extract metadata from sources (date, source credibility, topic focus): While this is less relevant as we manually curated our data, this could be useful when we adopt an agentic approach where the agent can browse the web for news sources.
- Create custom preprocessing for financial/economic text: Text cleaning can help to standardise the formatting differences across financial texts. Again, less of an issue if we use a bigger model, but a smaller model may struggle.
- Metadata Preservation in Chunks: This allows our model to correctly attribute the data source that it came from.

Our code for data processing is in `rag/document_processor.py` and the script for running it is in `scripts/ingest_documents.py`. Processed documents are stored in a json file in `data/processed`. Due to the smaller size of each article and the 128k context window of Ministral 3B, each article is just one chunk post-processing.