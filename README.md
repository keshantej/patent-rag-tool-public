# Patent RAG Tool

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A sophisticated Retrieval-Augmented Generation (RAG) system for patent law research and analysis. This tool leverages large language models to provide intelligent retrieval and contextualization across patent law's authoritative sources: the Manual of Patent Examining Procedure (MPEP), US Code Title 35 (USC), and Code of Federal Regulations Title 37 (CFR).

## 🚀 Features

- **Intelligent Document Retrieval**: Smart content routing across MPEP, USC, and CFR based on query type and legal context
- **Multi-Strategy Search**: Combines semantic search with direct section lookup and citation-based retrieval
- **Legal Query Processing**: Specialized query rewriting and classification for patent law terminology
- **Citation Linking**: Automatic hyperlink generation for legal references
- **Advanced Embedding System**: OpenAI-powered embeddings with intelligent caching for performance
- **Comprehensive Patent Law Coverage**: Complete MPEP chapters, 35 USC, and 37 CFR
- **Web Interface**: Simple browser-based UI for interactive queries

## 🔧 Technical Architecture

The Patent RAG Tool is built on a composable pipeline architecture using the Haystack framework, with several custom-built components:

- **Embedding Layer**: Dual-mode system supporting both OpenAI embeddings and local SentenceTransformers
- **Document Store**: Metadata-rich store with section titles, IDs, and source tracking
- **Query Pipeline**: Multi-stage processing with query classification, rewriting, and context enrichment
- **Retrieval System**: Hybrid retrieval combining BM25, dense embeddings, and legal structure awareness
- **Generation Layer**: OpenAI GPT models with patent-specific prompt engineering
- **Scraping Utilities**: Specialized scrapers for MPEP, USC, and CFR content

## 📊 Knowledge Base Coverage

The tool incorporates comprehensive patent law references:

- **MPEP**: All chapters (100-2900), covering patentability, prosecution, and procedures
- **Title 35 USC**: Complete patent statutes, including §101 (eligibility), §102 (novelty), §103 (non-obviousness)
- **Title 37 CFR**: Patent regulations covering rules of practice and examination
- **Special Rules**: Dedicated knowledge base for claim interpretation and counting rules

## 🛠️ Environment Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| MPEP_FOLDER_PATH | Path to the datapool directory | "datapool" |
| EMBEDDING_MODEL | SentenceTransformers model if OpenAI is not used | "BAAI/bge-base-en-v1.5" |
| OPENAI_EMBEDDING_MODEL | OpenAI embedding model | "text-embedding-3-small" |
| USE_OPENAI_EMBEDDINGS | Whether to use OpenAI embeddings | "true" |
| ANSWER_GENERATION_MODEL | OpenAI model for answer generation | "gpt-4.1" |
| QUERY_REWRITING_MODEL | OpenAI model for query rewriting | "o3" |
| EMBEDDING_CACHE_DIR | Directory for caching OpenAI embeddings | ".embedding_cache" |
| OPENAI_API_KEY | OpenAI API key | From auth.py |
| RETRIEVER_TOP_K | Number of documents to retrieve | 10 |
| CHUNKS_PER_DOC | Number of chunks to merge per document | 3 |

## 💻 Installation & Usage

### Prerequisites
- Python 3.9+
- OpenAI API key (optional, falls back to local models)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/patent-rag-tool.git
cd patent-rag-tool

# Install dependencies (requirements.txt not yet included, add as needed)
pip install haystack-ai openai sentence-transformers fastapi uvicorn nltk bs4
```

### Running the Application

```bash
# Run with default configuration
python run_pipeline.py

# Run with custom OpenAI settings
OPENAI_API_KEY=your_key USE_OPENAI_EMBEDDINGS=true python run_pipeline.py
```

### Data Collection

The tool includes scrapers for refreshing the knowledge base:

```bash
# Scrape the latest MPEP
python scraping/scrape_mpep.py

# Scrape 37 CFR (Patent Regulations)
python scraping/scrape_cfr37.py

# Scrape 35 USC (Patent Statutes)
python scraping/scrape_statutes.py
```

## 🔍 Use Cases

- **Patentability Research**: Quickly find relevant sections on eligibility, novelty, and obviousness
- **Procedural Guidance**: Identify MPEP procedures for specific examination scenarios
- **Legal Comparison**: Compare different sections of patent law for analysis
- **Case Analysis**: Apply patent law principles to specific fact patterns
- **Section Lookup**: Direct retrieval of specific MPEP sections, USC provisions, or CFR rules

## 🌟 Technical Highlights

- **Context-Aware Retrieval**: Intelligently determines which sources to search based on query analysis
- **Legal Domain Optimization**: Query rewriting specialized for patent terminology and references
- **Efficient Resource Usage**: Sophisticated caching to minimize API costs and improve response times
- **Adaptive Chunking**: Content segmentation that respects logical breaks while optimizing for context windows
- **Extensible Architecture**: Modular design allowing easy component replacement and enhancement

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.