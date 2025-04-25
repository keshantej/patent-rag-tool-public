# Patent RAG Tool (Public Demo Layer)

A Retrieval-Augmented Generation (RAG) platform purpose-built for U.S. patent law research and analysis.  
This system leverages large language models to provide contextualized, citation-linked answers using:

- The Manual of Patent Examining Procedure (MPEP)
- 35 U.S.C. (Patent Statute)
- 37 CFR (Patent Regulations)

## 🔐 Full Implementation

This public demo highlights the core architecture, components, and legal-tech design of the system.  
Proprietary routing logic, prompt assembly strategies, and document preprocessing pipelines are **retained in a private version** and available for **review upon request**.

## 🚀 Key Features (System Architecture)

The full system is designed to support:

- **Query Classification**: Identifies section lookups, comparisons, fact patterns, or claim-specific queries
- **Smart Retrieval Routing**: Dynamically switches between semantic and direct section retrieval
- **Statute-Aware Expansion**: If a query references 102 or 112, all relevant subsections and interpretations are added
- **Citation-Aware Prompt Building**: Contextualizes LLM output with inline [MPEP §], [CFR §], and [35 U.S.C. §] tags
- **Hyperlinking of Citations**: Automatically links citations in the output to authoritative sources
- **Legal Query Rewriting**: Refines user queries using LLMs to better match legal search indices
- **Efficient Embedding Layer**: Switchable between OpenAI or SentenceTransformers with optional caching
- **FastAPI Interface**: Lightweight API for legal question answering
- **Chunked Legal Vector Store**: In-memory vector database with metadata-rich sections

Some of these features are **stubbed or scaffolded** in the public version. Full functionality is available in the private edition.

## 📚 Legal Knowledge Base

Covers all major areas of U.S. patent law:

- **MPEP**: Chapters 100–2900 (Eligibility, Rejections, Appeal, Reissue, etc.)
- **Title 35 U.S.C.**: Statutory foundation for patent law (101, 102, 103, 112, etc.)
- **Title 37 CFR**: USPTO’s rules of practice and procedure
- **Claim Interpretation Rules**: Internal section dedicated to counting, dependencies, and fees

## ⚙️ Configuration

Configurable via environment variables (`.env` or shell):

| Variable | Description | Default |
|----------|-------------|---------|
| MPEP_FOLDER_PATH | Path to JSON chunk directory | `datapool/` |
| USE_OPENAI_EMBEDDINGS | Toggle OpenAI vs. local embeddings | `true` |
| EMBEDDING_MODEL | SentenceTransformers model | `BAAI/bge-base-en-v1.5` |
| OPENAI_EMBEDDING_MODEL | Embedding model name | `text-embedding-3-small` |
| ANSWER_GENERATION_MODEL | LLM for answering | `gpt-4.1` |
| QUERY_REWRITING_MODEL | LLM for query rewriting | `o3` |
| EMBEDDING_CACHE_DIR | Local cache path | `.embedding_cache/` |
| RETRIEVER_TOP_K | Number of top documents | `10` |
| CHUNKS_PER_DOC | Chunk merge size | `3` |

## 🧪 API Endpoint

- `POST /query`
- JSON payload:  
  ```json
  { "query": "difference between 102 and 103 rejections" }
  ```
- Response includes:  
  - Final answer with legal citations  
  - Source sections (with URLs, titles, and section IDs)

## 🛠️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/patent-rag-tool.git
cd patent-rag-tool

pip install -r requirements.txt
# or manually:
pip install haystack-ai openai sentence-transformers fastapi uvicorn nltk bs4
```

## ▶️ Running the Server

```bash
# With environment variables
OPENAI_API_KEY=your_key python public_run_pipeline.py
```

## 🛡 License

**© 2025 Rudra Tejiram. All rights reserved.**  
This public codebase is made available for **academic and evaluative purposes** only.  
**Reuse, modification, or redistribution is prohibited** without explicit written permission.

## 🤝 Contact & Licensing

To request full access, discuss collaborations, or licensing:

📧 rtejira1@jh.alumni.edu

---

*This project reflects an advanced legal AI system built for explainability, citation-grounding, and precision in U.S. patent law.*
