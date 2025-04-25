# Patent RAG Tool (Public Demo Layer)

A legal research tool for U.S. patent law, built with Retrieval-Augmented Generation (RAG) architecture.  
This demo showcases a simplified version of a system designed to deliver grounded, citation-rich answers using:

- The Manual of Patent Examining Procedure (MPEP)
- Title 35 U.S.C. (Patent Statutes)
- Title 37 CFR (Patent Regulations)

---

## 🔍 About This Project

This public repo includes the core architecture and components behind the tool.  
Some components—such as the full routing engine, dynamic prompt construction, and document preprocessing pipeline—are part of a more complete and private version of this tool. These features include statute-aware expansion, custom reranking logic, and citation-preserving chunking. While omitted from this public demo, they can be made available for academic evaluation, technical review, or collaborative discussions upon request.

---

## ⚙️ Key Features (Public Version)

- **Query Classification**: Distinguishes statute lookups, fact patterns, and claim-specific questions
- **Section-Aware Retrieval**: Switches between semantic and direct retrieval based on query type
- **Citation Tagging**: Adds inline references to [MPEP §], [CFR §], and [35 U.S.C. §]
- **Hyperlinked Citations**: Outputs include direct links to relevant legal sources
- **Basic Query Rewriting**: Uses LLMs to refine legal questions for better retrieval
- **Embedding Layer**: Supports OpenAI and SentenceTransformers with local caching
- **FastAPI Interface**: Simple legal QA endpoint
- **Web UI**: Tailwind-based frontend with Markdown rendering and citation memory

---

## 📚 Legal Corpus

- **MPEP**: Chapters 100–2900 (Eligibility, Appeals, Rejections, etc.)
- **Title 35**: Patent statutes including §§101, 102, 103, 112
- **Title 37 CFR**: USPTO’s rules of practice
- **Claim Rules & Fees**: Dedicated section for claims, dependencies, and filing costs

> The full version includes this corpus as well as a modular scraping tool in the case of updates. 
---

## 🧠 Embedding Logic

Includes a public version of the custom `OpenAIDocumentEmbedder`:

- Built with Haystack decorators
- Adds metadata like section titles and IDs
- Local hash-based caching to avoid duplicate API calls

> The full version includes reranker logic and additional retrieval layers.

---

## 🛠 Configuration

Set via `.env` or environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| MPEP_FOLDER_PATH | Path to chunked data | `datapool/` |
| USE_OPENAI_EMBEDDINGS | Toggle OpenAI embeddings | `true` |
| EMBEDDING_MODEL | SentenceTransformers model | `BAAI/bge-base-en-v1.5` |
| OPENAI_EMBEDDING_MODEL | OpenAI embed model | `text-embedding-3-small` |
| ANSWER_GENERATION_MODEL | Model for answers | `gpt-4.1` |
| QUERY_REWRITING_MODEL | Rewriter model | `o3` |
| EMBEDDING_CACHE_DIR | Cache folder | `.embedding_cache/` |
| RETRIEVER_TOP_K | Top documents retrieved | `10` |
| CHUNKS_PER_DOC | Chunk merge size | `3` |

---

## ▶️ API Usage

- **POST /query**
- Sample payload:
  ```json
  { "query": "difference between 102 and 103 rejections" }
  ```
- Response includes:
  - Answer with inline citations
  - Sources with URLs and section titles

---

## 🚀 Getting Started

```bash
git clone https://github.com/keshantej/patent-rag-tool-public.git
cd patent-rag-tool

pip install -r requirements.txt
# or manually:
pip install haystack-ai openai sentence-transformers fastapi uvicorn nltk bs4
```

### Run the Server

```bash
# With environment variables
OPENAI_API_KEY=your_key python public_run_pipeline.py
```

---

## 🛡 License & Use

**Certain components are proprietary and not included in this repo. Public content is licensed under the MIT License. For access to the full system or licensing inquiries, please contact me.**  
This codebase is shared for academic and demo purposes.

---

## 📬 Contact

For full access, licensing, or academic collaboration:  
📧 rtejira1@alumni.jh.edu

---

*This is an experimental legal tech tool built for transparency, explainability, and structured citation in patent law.*
