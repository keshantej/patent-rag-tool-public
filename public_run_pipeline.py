"""
Patent RAG Tool - Core Pipeline Implementation

This simplified version demonstrates the core retrieval-augmented generation (RAG) architecture
for a patent law system that leverages the MPEP, USC, and CFR as knowledge sources.

Key capabilities:
- Legal query classification and rewriting
- Context-aware document retrieval
- Hybrid search combining direct lookups and semantic search
- Dynamic document routing based on query type
- Statute-aware retrieval for legal completeness
- Citation linking with hyperlink generation

Note: This is a public, distilled version of the full system that highlights
the architecture and technical approach while omitting proprietary components.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import pickle

# --- Haystack Framework ---
from haystack import Pipeline, Document
from haystack.core.component import component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.utils.auth import Secret

# --- API Framework ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Custom Embedders ---
try:
    from embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
    OPENAI_EMBEDDERS_AVAILABLE = True
except ImportError:
    logging.warning("OpenAI embedders not available, using SentenceTransformers instead")
    OPENAI_EMBEDDERS_AVAILABLE = False

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Configuration ---
MPEP_FOLDER_PATH = Path(os.getenv("MPEP_FOLDER_PATH", "datapool"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "true").lower() in ("1", "true", "yes")
ANSWER_GENERATION_MODEL = os.getenv("ANSWER_GENERATION_MODEL", "gpt-4.1")
QUERY_REWRITING_MODEL = os.getenv("QUERY_REWRITING_MODEL", "o3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", ".embedding_cache")
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 10))
CHUNKS_PER_DOC = int(os.getenv("CHUNKS_PER_DOC", "3"))

# --- Core Components ---

@component
class MPEPQueryClassifier:
    """
    Classifies patent law queries by type to optimize retrieval strategy:
    - Section queries: Direct MPEP/USC/CFR section lookups
    - Comparative queries: Legal comparison across multiple sections
    - Claim counting: Special handling for patent claim rules
    - Fact patterns: Broad retrieval for applied legal analysis
    - General queries: Standard semantic search
    """
    @component.output_types(
        is_section_query=bool,
        section_id=str,
        original_query=str,
        question_type=str,
        comparison_entities=List[str]
    )
    def run(self, query: str):
        orig = query
        section_patterns = [
            r"\b(mpep|usc|cfr)\s*[\u00a7§]?\s*([0-9]{2,5}(?:\.[0-9A-Za-z]+)*(?:\s*\([a-z0-9]\))*)\b",
            r"\b(section|§)\s*([0-9]{3,5}(?:\.[0-9A-Za-z]+)*)\b"
        ]
        comparison_keywords = ["difference", "different", "compare", "vs", "versus"]
        fact_keywords = ["claims", "invention", "reference", "examiner", "file", "argues", "reject", "prior art"]
        claim_pattern = r"claim\s+\d+"
        claim_counting_keywords = [
            "multiple dependent", "dependent claim", "independent claim", 
            "claim fee", "claim count", "counting claims", "claim dependency"
        ]

        # 1. Section Query Check
        is_section_query = False
        section_id = ""
        for pat in section_patterns:
            m = re.search(pat, query, re.I)
            if m:
                is_section_query = True
                section_id = m.group(2).strip()
                break

        # 2. Determine Query Type
        question_type = "general"
        comparison_entities = []
        
        # Claim Counting Check
        if re.search(claim_pattern, query.lower()) or any(kw in query.lower() for kw in claim_counting_keywords):
            question_type = "claim_counting"
            if not is_section_query:
                section_id = "claim_counting_rules"
                is_section_query = True
                
        # Comparative Check
        elif any(kw in query.lower() for kw in comparison_keywords) and re.findall(r"1[0-9]{2}|102|103|112|101", query):
            question_type = "comparative"
            comparison_entities = list(set(re.findall(r"(1[0-9]{2}|101|102|103|112)", query)))
            
        # Fact Pattern Check
        elif len(re.split(r'[.!?]', query)) > 2 and any(kw in query.lower() for kw in fact_keywords):
            question_type = "fact_pattern"
            
        # Default to section query if section was identified
        elif is_section_query:
            question_type = "section"

        return {
            "is_section_query": is_section_query,
            "section_id": section_id,
            "original_query": orig,
            "question_type": question_type,
            "comparison_entities": comparison_entities
        }

@component
class DirectSectionRetriever:
    """
    Handles direct retrieval of legal sections by ID, with intelligent expansion:
    - Exact match: Returns exact section (e.g., "102(a)(1)")
    - Subsection handling: For section 102(a), also retrieves 102(b) and parent section 102
    - Parent/child navigation: For section lookups, retrieves related subsections
    """
    def __init__(self, document_store: InMemoryDocumentStore):
        self.document_store = document_store

    @component.output_types(documents=List[Document])
    def run(self, section_id: str):
        all_docs = self.document_store.filter_documents({})
        # 1. Exact match lookup
        exact = [doc for doc in all_docs if doc.meta.get("section_id") == section_id]

        # 2. Sibling/adjacent section retrieval
        sibling_docs = []
        m = re.match(r'^([\d.]+)\(([a-z])\)$', section_id)
        if m:
            base = m.group(1)  # e.g., 706.07
            letter = m.group(2) # e.g., "a"
            next_letter = chr(ord(letter) + 1)
            prev_letter = chr(ord(letter) - 1) if letter > 'a' else None

            # Try next and previous sibling if they exist
            for ltr in filter(None, [prev_letter, next_letter]):
                sibling_id = f"{base}({ltr})"
                sibs = [doc for doc in all_docs if doc.meta.get("section_id") == sibling_id]
                sibling_docs.extend(sibs)
            # Also add the base section
            parent_docs = [doc for doc in all_docs if doc.meta.get("section_id") == base]
            sibling_docs.extend(parent_docs)
        
        # 3. Find subsections for a parent section
        prefix = []
        if not exact:
            for doc in all_docs:
                sid = doc.meta.get("section_id", "")
                if sid.startswith(section_id + ".") or sid.startswith(section_id + "("):
                    prefix.append(doc)
        
        # 4. Find parent sections if needed
        parent = []
        if not exact and not prefix:
            pieces = re.split(r'[.(]', section_id)
            for i in range(len(pieces)-1, 0, -1):
                parent_sid = '.'.join(pieces[:i])
                for doc in all_docs:
                    if doc.meta.get("section_id") == parent_sid:
                        parent.append(doc)

        # Combine and deduplicate results
        results = exact + sibling_docs
        if not results:
            results = exact or sibling_docs or prefix or parent

        # Remove duplicates by section_id
        deduped = {}
        for doc in results:
            deduped[doc.meta.get("section_id")] = doc
        return {"documents": list(deduped.values())[:12]}

@component
class SmartRouter:
    """
    Intelligent document routing system that:
    1. For comparative questions (e.g., "102 vs. 103"), ensures all relevant sections are included
    2. For claim counting questions, prioritizes claim counting rules
    3. For section queries, relies on direct retrieval
    4. For general queries, uses semantic retrieval
    """
    @component.output_types(
        documents=List[Document],
        query=str,
        question_type=str,
        entities=List[str]
    )
    def run(
        self,
        is_section_query: bool,
        section_documents: Optional[List[Document]] = None,
        semantic_documents: Optional[List[Document]] = None,
        original_query: str = "",
        rewritten_query: Optional[str] = None,
        question_type: str = "general",
        comparison_entities: Optional[List[str]] = None
    ):
        section_docs = section_documents if section_documents else []
        semantic_docs = semantic_documents if semantic_documents else []
        entities = comparison_entities or []

        # Comparative mode: select best per entity
        if question_type == "comparative" and entities:
            logger.info(f"Router: comparative - entities {entities}")
            # For each entity, select (from available docs)
            selected = []
            docs_by_sid = {}
            for doc in section_docs + semantic_docs:
                sid = doc.meta.get("section_id", "")
                for ent in entities:
                    if ent in sid:
                        docs_by_sid.setdefault(ent, []).append(doc)
            for ent in entities:
                if docs_by_sid.get(ent):
                    selected.append(docs_by_sid[ent][0])
            documents = selected
            
        # Claim counting handling
        elif question_type == "claim_counting":
            logger.info("Router: claim_counting detected")
            # Ensure claim_counting_rules is in docs, prioritize it
            claim_counting_docs = []
            other_docs = []
            
            # First check section_docs (from direct retrieval)
            for doc in section_docs:
                if doc.meta.get("section_id") == "claim_counting_rules":
                    claim_counting_docs.append(doc)
                else:
                    other_docs.append(doc)
                    
            # If not found in section_docs, check semantic_docs
            if not claim_counting_docs:
                for doc in semantic_docs:
                    if doc.meta.get("section_id") == "claim_counting_rules":
                        claim_counting_docs.append(doc)
                    else:
                        other_docs.append(doc)
            
            # Always prioritize claim_counting_rules docs first, then add other relevant docs
            documents = claim_counting_docs + other_docs
            
        # Direct section lookup
        elif is_section_query and section_docs:
            documents = section_docs
            
        # Default to semantic search
        elif semantic_docs:
            documents = semantic_docs
            
        # Fallback if nothing found
        else:
            documents = []

        return {
            "documents": documents,
            "query": original_query,
            "question_type": question_type,
            "entities": entities
        }

@component
class StatuteAwareRetriever:
    """
    Legal domain-specific retriever that ensures comprehensive coverage of related statutes:
    - For 102/103 questions, ensures all subsections are included
    - Enhances retrieval with relevant MPEP analysis sections
    - Ensures complete coverage of legal concepts
    """
    def __init__(self, document_store: InMemoryDocumentStore):
        self.document_store = document_store

        # Key statute mappings to related sections
        self.STATUTE_MAPPINGS = {
            "pre-aia 102": {
                "subsections": [
                    "102(a)", "102(b)", "102(c)", "102(d)", "102(e)", "102(f)", "102(g)", "102"
                ],
                "mpep_sections": ["2131", "2132", "2133", "2134", "2135", "2136", "2137", "2138"]
            },
            "aia 102": {
                "subsections": [
                    "102(a)(1)", "102(a)(2)", "102(b)(1)", "102(b)(2)", "102(c)", "102(d)", "102"
                ],
                "mpep_sections": ["2150", "2151", "2152", "2153", "2154"]
            },
            "103": {
                "subsections": ["103(a)", "103(b)", "103(c)", "103"],
                "mpep_sections": ["2141", "2142", "2143", "2144", "2145"]
            },
            "101": {
                "subsections": ["101"],
                "mpep_sections": ["2106", "2107", "2105"]
            },
            "claim_counting_rules": {
                "subsections": ["claim_counting_rules"],
                "mpep_sections": ["608.01(n)", "714.10"]  # MPEP sections related to claim counting
            }
        }

    def _find_relevant_statute_family(self, query: str) -> Optional[str]:
        """Determine if the query asks about a broad statute/subsection."""
        q = query.lower()
        # Look for statute references and claim counting patterns
        claim_pattern = r"claim\s+\d+"
        claim_counting_keywords = [
            "multiple dependent", "dependent claim", "independent claim", 
            "claim fee", "claim count", "counting claims", "claim dependency"
        ]
        
        # Check for claim counting patterns
        if (re.search(claim_pattern, q) or any(keyword in q for keyword in claim_counting_keywords)):
            return "claim_counting_rules"
            
        # Check for statute references
        if ("102" in q and not re.search(r"102\([a-z]\)", q)):
            return "pre-aia 102"
        if "103" in q:
            return "103"
        if "101" in q:
            return "101"
        
        return None

    def _fetch_by_section(self, sections: list) -> list:
        """Retrieve all documents with a matching section_id."""
        all_docs = self.document_store.filter_documents({})
        docs = []
        for sec in sections:
            docs.extend([doc for doc in all_docs if doc.meta.get("section_id", "") == sec])
        return docs

    @component.output_types(documents=List[Document])
    def run(self, 
            original_query: str,
            retrieved_documents: Optional[List[Document]] = None
            ):
        """
        Enhances retrieval by adding related statute sections and MPEP chapters.
        """
        key_statute = self._find_relevant_statute_family(original_query)
        extra_docs = []
        if key_statute and key_statute in self.STATUTE_MAPPINGS:
            mapping = self.STATUTE_MAPPINGS[key_statute]
            # Fetch all related statute sections
            extra_docs += self._fetch_by_section(mapping["subsections"])
            # Fetch related MPEP sections
            extra_docs += self._fetch_by_section(mapping["mpep_sections"])
            
        # Merge with initial retrieval docs, deduplicate
        if not retrieved_documents:
            combined = extra_docs
        else:
            # Union and dedupe by doc.meta["section_id"]
            seen = set()
            combined = []
            for d in (retrieved_documents + extra_docs):
                sid = d.meta.get("section_id")
                if sid and sid not in seen:
                    seen.add(sid)
                    combined.append(d)
        
        return {"documents": combined[:15]}

@component
class CitationLinker:
    """
    Enhances output by converting legal citations to hyperlinks:
    - Detects citations like [MPEP § 2131], [35 U.S.C. § 102], [37 CFR § 1.121]
    - Links them to the actual document sources
    - Improves navigation by allowing users to click through to referenced sections
    """
    def __init__(self, document_store: InMemoryDocumentStore):
        # Build a mapping from section_id to URL for quick lookup
        self._url_map: Dict[str, str] = {}
        try:
            all_docs = document_store.filter_documents({})
            for doc in all_docs:
                sid = doc.meta.get("section_id")
                url = doc.meta.get("url")
                if sid and url:
                    # Store the exact section_id -> url
                    self._url_map[sid] = url
        except Exception:
            logger.warning("CitationLinker: failed to build URL map from document store", exc_info=True)

    @component.output_types(linked_answer=str)
    def run(self, replies: List[str]):
        # Get the LLM's answer text
        answer = replies[0] if replies and isinstance(replies, list) else ""
        # Regex for bracketed blocks
        bracket_pattern = re.compile(r"\[([^\]]+)\]")
        # Regex to identify individual citations inside a bracket
        citation_pattern = re.compile(r"(?P<prefix>MPEP|35\s*U\.S\.C\.|37\s*CFR)\s*§\s*(?P<sid>[0-9A-Za-z.()]+)")
        
        def replace_bracket(match):
            content = match.group(1)
            # Within this bracket, replace each citation separately
            def replace_cite(cit):
                original = cit.group(0)
                sid = cit.group('sid').strip()
                url = self._url_map.get(sid)
                if url:
                    return f"[{original}]({url})"
                return original
            # Apply citation-level substitution, preserving commas and whitespace
            new_content = citation_pattern.sub(replace_cite, content)
            return f"[{new_content}]"
            
        # Process all bracketed groups in the answer
        linked = bracket_pattern.sub(replace_bracket, answer)
        return {"linked_answer": linked}

# --- Helper Functions ---

def load_mpep_json_files(folder_path: Path) -> List[Document]:
    """
    Loads legal documents from JSON files with:
    - Chunking logic to optimize context windows
    - Metadata preservation for section IDs, titles, and URLs
    - Error handling for malformed files
    """
    if not folder_path.is_dir():
        logger.error(f"Legal document folder not found: {folder_path}")
        return []
        
    logger.info(f"Loading legal documents from: {folder_path}")
    sections: Dict[str, Dict[str, Any]] = {}
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".json"):
            continue
            
        full_path = folder_path / filename
        try:
            with open(full_path, "r", encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            continue
            
        if not isinstance(data, list):
            logger.warning(f"Skipping non-list JSON file: {filename}")
            continue
            
        for item in data:
            if not all(k in item for k in ("text", "section_id", "title", "url", "chunk_id")):
                continue
                
            text = item.get("text", "").strip()
            if not text:
                continue
                
            sid = item["section_id"]
            # initialize section entry
            if sid not in sections:
                sections[sid] = {"title": item.get("title", ""),
                              "url": item.get("url", ""),
                              "chunks": []}
            # collect chunk (id, text)
            sections[sid]["chunks"].append((str(item.get("chunk_id", "0")), text))
            
    # Build documents: merge original JSON chunks into smaller documents
    documents: List[Document] = []
    for sid, info in sections.items():
        # Sort chunks by chunk_id (numeric then lexicographic)
        nums = []  # type: List[tuple[int, str]]
        strs = []  # type: List[tuple[str, str]]
        for cid, txt in info["chunks"]:
            if cid.isdigit():
                nums.append((int(cid), txt))
            else:
                strs.append((cid, txt))
        nums.sort(key=lambda x: x[0])
        strs.sort(key=lambda x: x[0])
        ordered_chunks = [(str(cid), txt) for cid, txt in nums] + strs
        
        # Group and merge chunks into documents of size CHUNKS_PER_DOC
        for i in range(0, len(ordered_chunks), CHUNKS_PER_DOC):
            group = ordered_chunks[i:i + CHUNKS_PER_DOC]
            texts = [txt for _, txt in group]
            merged_content = "\n\n".join(texts)
            meta = {
                "section_id": sid,
                "title": info.get("title", ""),
                "url": info.get("url", ""),
                "chunk_ids": [cid for cid, _ in group]
            }
            documents.append(Document(content=merged_content, meta=meta))
            
    logger.info(f"Loaded and merged {len(documents)} documents (chunks per doc={CHUNKS_PER_DOC})")
    return documents

def build_and_run_indexing_pipeline(docs: List[Document], doc_store: InMemoryDocumentStore):
    """Creates and executes the document indexing pipeline with embedding generation"""
    if not docs:
        logger.warning("Indexing skipped: No documents given.")
        return
        
    logger.info(f"Indexing {len(docs)} documents...")
    
    # Select embedder based on configuration
    if USE_OPENAI_EMBEDDINGS and OPENAI_EMBEDDERS_AVAILABLE:
        logger.info(f"Using OpenAI embeddings with model: {OPENAI_EMBEDDING_MODEL}")
        doc_embedder = OpenAIDocumentEmbedder(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_EMBEDDING_MODEL,
            meta_fields_to_embed=["title", "section_id"],
            embedding_cache_dir=EMBEDDING_CACHE_DIR
        )
    else:
        logger.info(f"Using SentenceTransformers embeddings with model: {EMBEDDING_MODEL}")
        doc_embedder = SentenceTransformersDocumentEmbedder(
            model=EMBEDDING_MODEL,
            meta_fields_to_embed=["title", "section_id"]
        )
        doc_embedder.warm_up()

    writer = DocumentWriter(document_store=doc_store, policy="overwrite")
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", doc_embedder)
    indexing_pipeline.add_component("writer", writer)
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    
    try:
        indexing_pipeline.run({"embedder": {"documents": docs}})
        logger.info(f"Successfully indexed {doc_store.count_documents()} documents.")
    except Exception as e:
        logger.error(f"Error in indexing pipeline: {e}", exc_info=True)
        raise

def build_query_pipeline(doc_store: InMemoryDocumentStore) -> Pipeline:
    """
    Builds a comprehensive RAG pipeline with:
    - Query classification for context-aware retrieval
    - Query rewriting for better semantic matching
    - Hybrid retrieval combining direct lookups and embedding similarity
    - Patent law-specific prompt engineering
    """
    logger.info("Building hybrid RAG query pipeline...")

    # --- Core Components ---
    query_classifier = MPEPQueryClassifier()
    direct_retriever = DirectSectionRetriever(document_store=doc_store)
    
    # Query rewriting prompt 
    rewrite_template = """
    You are a legal research assistant specializing in U.S. patent law.

    Your task is to rewrite the user's question to improve legal retrieval and coverage, particularly when searching the MPEP, the U.S. Code (U.S.C.), or the Code of Federal Regulations (CFR). 
    
    - Expand statute references (e.g., "102" → "35 U.S.C. § 102")
    - Include relevant legal terminology and synonyms
    - For fact patterns, identify key legal issues
    - For comparative questions, clearly identify all sections being compared
    - Preserve technical language and legal references

    Only rewrite the query — do not answer the question.

    User Question: {{ query }}

    Rewritten Query:
    """
    
    rewriter_prompt_builder = PromptBuilder(template=rewrite_template, required_variables=["query"])
    query_rewriter_llm = OpenAIGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model=QUERY_REWRITING_MODEL)
    
    # Extract the rewritten query
    @component
    class ReplyExtractor:
        @component.output_types(text=str)
        def run(self, replies: List[str]):
            if not replies or not isinstance(replies, list):
                return {"text": ""}
            first_reply = replies[0] if replies else ""
            if not isinstance(first_reply, str):
                return {"text": ""}
            return {"text": first_reply}
            
    reply_extractor = ReplyExtractor()

    # Embedding components
    if USE_OPENAI_EMBEDDINGS and OPENAI_EMBEDDERS_AVAILABLE:
        logger.info(f"Using OpenAI query embeddings with model: {OPENAI_EMBEDDING_MODEL}")
        query_embedder = OpenAITextEmbedder(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_EMBEDDING_MODEL,
            embedding_cache_dir=EMBEDDING_CACHE_DIR
        )
    else:
        logger.info(f"Using SentenceTransformers query embeddings with model: {EMBEDDING_MODEL}")
        query_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)
        query_embedder.warm_up()
    
    # Vector retrieval and re-ranking
    retriever = InMemoryEmbeddingRetriever(
        document_store=doc_store,
        top_k=RETRIEVER_TOP_K
    )
    
    from haystack.components.rankers import TransformersSimilarityRanker
    reranker = TransformersSimilarityRanker(
        model="BAAI/bge-reranker-large",
        top_k=13,
    )

    # Smart routing and statute-aware retrieval
    router = SmartRouter()
    statute_aware_retriever = StatuteAwareRetriever(document_store=doc_store)
    
    # Document length management
    @component
    class DocumentTokenLimiter:
        """Limits the total tokens in documents to fit within LLM context window."""
        def __init__(self, max_tokens: int = 60000):
            self.max_tokens = max_tokens
            self.reserved_tokens = 4000
            self.available_tokens = max_tokens - self.reserved_tokens
            
        @component.output_types(documents=List[Document])
        def run(self, documents: List[Document]):
            """Limits document content to fit within token budget."""
            if not documents:
                return {"documents": []}
                
            # Rough token count estimation (4 chars ≈ 1 token)
            total_tokens = sum(len(doc.content) // 4 for doc in documents)
            
            if total_tokens <= self.available_tokens:
                return {"documents": documents}
            
            # We need to reduce the documents
            logger.info(f"Truncating documents to fit {self.available_tokens} token limit")
            
            # Sort documents by relevance/importance
            sorted_docs = documents.copy()
            
            # Allocate tokens to prioritize first few documents
            num_priority_docs = min(5, len(sorted_docs))
            priority_allocation = int(self.available_tokens * 0.75)
            remaining_allocation = self.available_tokens - priority_allocation
            
            # Process documents for token budget
            result_docs = []
            allocated_tokens = 0
            
            # Process priority documents
            for i, doc in enumerate(sorted_docs[:num_priority_docs]):
                doc_tokens = len(doc.content) // 4
                max_doc_tokens = max(1000, priority_allocation // num_priority_docs)
                
                if doc_tokens <= max_doc_tokens:
                    result_docs.append(doc)
                    allocated_tokens += doc_tokens
                else:
                    # Truncate document if needed
                    truncated_length = max_doc_tokens * 4
                    truncated_content = doc.content[:truncated_length] + "... [truncated]"
                    truncated_doc = Document(
                        content=truncated_content,
                        meta=doc.meta.copy()
                    )
                    result_docs.append(truncated_doc)
                    allocated_tokens += max_doc_tokens
            
            # Process remaining documents if space allows
            remaining_docs = sorted_docs[num_priority_docs:]
            if remaining_docs and (allocated_tokens < self.available_tokens):
                tokens_per_doc = remaining_budget = self.available_tokens - allocated_tokens
                if len(remaining_docs) > 0:
                    tokens_per_doc = remaining_budget // len(remaining_docs)
                
                for doc in remaining_docs:
                    doc_tokens = len(doc.content) // 4
                    if doc_tokens <= tokens_per_doc:
                        result_docs.append(doc)
                        allocated_tokens += doc_tokens
                    elif tokens_per_doc > 200:  # Only include if we can keep a meaningful amount
                        truncated_length = tokens_per_doc * 4
                        truncated_content = doc.content[:truncated_length] + "... [truncated]"
                        truncated_doc = Document(
                            content=truncated_content,
                            meta=doc.meta.copy()
                        )
                        result_docs.append(truncated_doc)
                        allocated_tokens += tokens_per_doc
            
            return {"documents": result_docs}
    
    token_limiter = DocumentTokenLimiter()
    
    # Final answer generation. This is a sample prompt that outlines the general goal.
    final_prompt_template = """
    You are an experienced patent attorney and patent law professor. 
    
    Your task is to answer the user's question intelligently, grounded in U.S. patent law, primarily using the provided 'Context Sections' from MPEP, CFR, and USC.

    Your response must:

    - Be comprehensive yet concise
    - Maintain legal accuracy and precision
    - Include direct quotes from the provided context using > format
    - Use professional formatting with headings, lists, bullet points, or tables
    - Cite sources properly: [MPEP § XXXX], [37 CFR § X.XXX], [35 U.S.C. § XXX]
    - Distinguish between pre-AIA and post-AIA provisions when relevant

    ---
    ## Context to use:

    {% for doc in documents %}
    Section ID: [{{ doc.meta.section_id }}]
    Title: {{ doc.meta.title }}
    Content:
    {{ doc.content }}

    ---
    {% endfor %}

    Original User Question: {{ query }}

    Answer:
    """
    
    final_prompt_builder = PromptBuilder(template=final_prompt_template, required_variables=["query", "documents"])
    
    generator_kwargs = {
        "api_key": Secret.from_token(OPENAI_API_KEY),
        "model": ANSWER_GENERATION_MODEL,
        "timeout": 40
    }
    
    # Add generation_kwargs if supported by OpenAIGenerator
    try:
        from inspect import signature
        if "generation_kwargs" in signature(OpenAIGenerator.__init__).parameters:
            generator_kwargs["generation_kwargs"] = {"max_tokens": 4000}
    except:
        pass
        
    answer_generator_llm = OpenAIGenerator(**generator_kwargs)
    
    # Citation linking for improved navigation
    citation_linker = CitationLinker(document_store=doc_store)

    # --- Build Pipeline ---
    query_pipeline = Pipeline()
    
    # Add components
    query_pipeline.add_component("query_classifier", query_classifier)
    query_pipeline.add_component("direct_retriever", direct_retriever)
    query_pipeline.add_component("rewriter_prompt_builder", rewriter_prompt_builder)
    query_pipeline.add_component("query_rewriter_llm", query_rewriter_llm)
    query_pipeline.add_component("reply_extractor", reply_extractor)
    query_pipeline.add_component("query_embedder", query_embedder)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.add_component("reranker", reranker)
    query_pipeline.add_component("router", router)
    query_pipeline.add_component("statute_aware_retriever", statute_aware_retriever)
    query_pipeline.add_component("token_limiter", token_limiter)
    query_pipeline.add_component("final_prompt_builder", final_prompt_builder)
    query_pipeline.add_component("answer_generator_llm", answer_generator_llm)
    query_pipeline.add_component("citation_linker", citation_linker)

    # --- Connect components ---
    query_pipeline.connect("query_classifier.is_section_query", "router.is_section_query")
    query_pipeline.connect("query_classifier.section_id", "direct_retriever.section_id")
    query_pipeline.connect("direct_retriever.documents", "router.section_documents")
    query_pipeline.connect("query_classifier.original_query", "router.original_query")
    query_pipeline.connect("query_classifier.question_type", "router.question_type")
    query_pipeline.connect("query_classifier.comparison_entities", "router.comparison_entities")

    query_pipeline.connect("query_classifier.original_query", "rewriter_prompt_builder.query")
    query_pipeline.connect("rewriter_prompt_builder.prompt", "query_rewriter_llm.prompt")
    query_pipeline.connect("query_rewriter_llm.replies", "reply_extractor.replies")
    query_pipeline.connect("reply_extractor.text", "router.rewritten_query")
    query_pipeline.connect("reply_extractor.text", "query_embedder.text")
    query_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever.documents", "reranker.documents")
    query_pipeline.connect("reply_extractor.text", "reranker.query")
    query_pipeline.connect("reranker.documents", "router.semantic_documents")

    # Connect router to statute-aware retriever and document processing
    query_pipeline.connect("router.documents", "statute_aware_retriever.retrieved_documents")
    query_pipeline.connect("router.query", "statute_aware_retriever.original_query")
    query_pipeline.connect("statute_aware_retriever.documents", "token_limiter.documents")
    query_pipeline.connect("token_limiter.documents", "final_prompt_builder.documents")
    query_pipeline.connect("router.query", "final_prompt_builder.query")
    query_pipeline.connect("final_prompt_builder.prompt", "answer_generator_llm.prompt")
    query_pipeline.connect("answer_generator_llm.replies", "citation_linker.replies")

    logger.info("RAG pipeline built successfully")
    return query_pipeline

# --- Document Store and Pipeline Setup ---
def load_document_store():
    """Creates or loads document store from cache"""
    cache_dir = Path(os.getenv("CACHE_DIR", ".cache"))
    cache_dir.mkdir(exist_ok=True)
    
    # Create embedding cache directory
    Path(EMBEDDING_CACHE_DIR).mkdir(exist_ok=True, parents=True)
    
    index_store_file = cache_dir / "doc_store.pkl"
    
    document_store = None
    if index_store_file.exists():
        try:
            logger.info(f"Loading DocumentStore from cache")
            with open(index_store_file, "rb") as f:
                document_store = pickle.load(f)
                
            # Check if store has documents
            doc_count = document_store.count_documents()
            if doc_count == 0:
                logger.warning("Cached DocumentStore is empty, rebuilding...")
                document_store = None
            else:
                logger.info(f"Loaded DocumentStore with {doc_count} documents")
        except Exception as e:
            logger.warning(f"Failed to load cached DocumentStore: {e}")
            document_store = None
            
    if document_store is None:
        # Create new document store and index documents
        document_store = InMemoryDocumentStore(embedding_similarity_function="dot_product")
        mpep_documents = load_mpep_json_files(MPEP_FOLDER_PATH)
        if mpep_documents:
            build_and_run_indexing_pipeline(mpep_documents, document_store)
            
            # Save to cache
            try:
                with open(index_store_file, "wb") as f:
                    pickle.dump(document_store, f)
                logger.info(f"Saved DocumentStore to cache")
            except Exception as e:
                logger.warning(f"Failed to cache DocumentStore: {e}")
                
    return document_store

# --- FastAPI Setup ---
app = FastAPI(
    title="Proof of Concept Patent Law RAG API",
    description="API for querying patent law documents using retrieval-augmented generation. Built by Rudra Tejiram (rtejira1@alumni.jh.edu)",
    version="1.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The user's legal question for the system.")

class SourceDocument(BaseModel):
    section_id: str
    title: str
    url: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    original_query: str

# --- API Endpoints ---
@app.get("/", summary="Health check")
async def root():
    """Health check endpoint for root path."""
    return {"status": "Patent Law RAG API is running"}

@app.post("/query", response_model=QueryResponse)
async def query_api(request: QueryRequest):
    """
    Processes a legal query through the RAG pipeline and returns a comprehensive answer
    with citations and references to relevant legal sources.
    """
    query_text = request.query
    logger.info(f"Received query: {query_text}")

    # Initialize document store and pipeline on first request
    global document_store, rag_pipeline
    if 'document_store' not in globals() or document_store is None:
        document_store = load_document_store()
        
    if 'rag_pipeline' not in globals() or rag_pipeline is None:
        if document_store and document_store.count_documents() > 0:
            rag_pipeline = build_query_pipeline(document_store)
        else:
            raise HTTPException(status_code=503, detail="Service unavailable: Document store not initialized")

    try:
        # Run the query through the pipeline
        result = rag_pipeline.run(
            {"query_classifier": {"query": query_text}},
            include_outputs_from=[
                "citation_linker",
                "router",
                "answer_generator_llm"
            ]
        )

        # Extract the answer with hyperlinked citations if available
        if "citation_linker" in result and result["citation_linker"].get("linked_answer"):
            answer = result["citation_linker"]["linked_answer"].strip()
        else:
            answer = result["answer_generator_llm"]["replies"][0].strip()

        # Extract sources from the router component
        if "router" in result and "documents" in result["router"]:
            sources = [
                SourceDocument(
                    section_id=doc.meta.get("section_id", "N/A"),
                    title=doc.meta.get("title", "N/A"),
                    url=doc.meta.get("url", "#")
                )
                for doc in result["router"]["documents"]
            ]
        else:
            sources = []

        return QueryResponse(
            answer=answer,
            sources=sources,
            original_query=query_text
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing your query: {str(e)}")

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Initializing document store...")
    document_store = load_document_store()
    
    if document_store.count_documents() > 0:
        logger.info("Building RAG pipeline...")
        rag_pipeline = build_query_pipeline(document_store)
        logger.info("Starting FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=7128)
    else:
        logger.critical("Failed to initialize document store. Exiting.")
        exit(1)