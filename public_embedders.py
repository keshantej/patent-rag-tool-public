"""
OpenAI Embedders for Haystack with Persistence

This module provides OpenAI-powered embedding components that serve as
drop-in replacements for SentenceTransformers embedders in Haystack,
while adding persistence capabilities to reduce API costs.

Components:
- OpenAIDocumentEmbedder: Embeds documents with metadata-aware capabilities
- OpenAITextEmbedder: Embeds query texts with caching for common queries
- EmbeddingCache: Handles caching of embeddings to disk

These components enhance the standard Haystack pipeline by reducing
redundant API calls while ensuring embedding quality.
"""

import os
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
from haystack import Document, component
from haystack.utils.auth import Secret
from openai import OpenAI

# Configure logger
logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    Simple cache for OpenAI embeddings that stores document and query embeddings
    to avoid redundant API calls.
    """
    
    def __init__(self, cache_dir: str = ".embedding_cache"):
        """Initialize the embedding cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # In-memory caches
        self.doc_cache: Dict[str, Dict[str, Any]] = {}
        self.query_cache: Dict[str, np.ndarray] = {}
        
        # Load existing cache
        self._load_cache()
        
    def _load_cache(self) -> None:
        """Load cache from disk if available."""
        doc_cache_path = self.cache_dir / "document_embeddings.pkl"
        query_cache_path = self.cache_dir / "query_embeddings.pkl"
        
        if doc_cache_path.exists():
            try:
                with open(doc_cache_path, "rb") as f:
                    self.doc_cache = pickle.load(f)
                logger.info(f"Loaded document embedding cache with {len(self.doc_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load document embedding cache: {e}")
                self.doc_cache = {}
        
        if query_cache_path.exists():
            try:
                with open(query_cache_path, "rb") as f:
                    self.query_cache = pickle.load(f)
                logger.info(f"Loaded query embedding cache with {len(self.query_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load query embedding cache: {e}")
                self.query_cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        doc_cache_path = self.cache_dir / "document_embeddings.pkl"
        query_cache_path = self.cache_dir / "query_embeddings.pkl"
        
        try:
            with open(doc_cache_path, "wb") as f:
                pickle.dump(self.doc_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save document embedding cache: {e}")
        
        try:
            with open(query_cache_path, "wb") as f:
                pickle.dump(self.query_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save query embedding cache: {e}")
    
    def get_doc_embedding(self, doc_id: str, content_hash: str) -> Optional[np.ndarray]:
        """
        Get document embedding from cache if available and not changed.
        
        Args:
            doc_id: Document ID
            content_hash: Hash of document content to detect changes
            
        Returns:
            Cached embedding if found and content unchanged, None otherwise
        """
        if doc_id in self.doc_cache:
            if self.doc_cache[doc_id]["content_hash"] == content_hash:
                return np.array(self.doc_cache[doc_id]["embedding"])
        return None
    
    def save_doc_embedding(self, doc_id: str, content_hash: str, embedding: np.ndarray) -> None:
        """
        Save document embedding to cache.
        
        Args:
            doc_id: Document ID
            content_hash: Hash of document content
            embedding: The embedding to cache
        """
        self.doc_cache[doc_id] = {
            "content_hash": content_hash,
            "embedding": embedding.tolist()  # Convert to list for serialization
        }
        # Periodically save cache
        if len(self.doc_cache) % 10 == 0:
            self._save_cache()
    
    def get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Get query embedding from cache.
        
        Args:
            query: The query text
            
        Returns:
            Cached embedding if found, None otherwise
        """
        query_hash = self._hash_text(query)
        if query_hash in self.query_cache:
            return self.query_cache[query_hash]
        return None
    
    def save_query_embedding(self, query: str, embedding: np.ndarray) -> None:
        """
        Save query embedding to cache.
        
        Args:
            query: The query text
            embedding: The embedding to cache
        """
        query_hash = self._hash_text(query)
        self.query_cache[query_hash] = embedding
        # Save cache periodically
        if len(self.query_cache) % 10 == 0:
            self._save_cache()
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Create a hash of text content."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    @staticmethod
    def _hash_document(doc: Document) -> str:
        """Create a hash of document content and relevant metadata."""
        content = doc.content
        metadata_str = ""
        if doc.meta:
            # Include relevant metadata that affects embedding
            for key in sorted(doc.meta.keys()):
                if key in ["title", "section_id"]:
                    metadata_str += f"{key}:{doc.meta[key]},"
        
        combined = f"{content}|{metadata_str}"
        return hashlib.md5(combined.encode("utf-8")).hexdigest()


@component
class OpenAIDocumentEmbedder:
    """
    A Haystack component that embeds Documents using OpenAI's embedding models.
    
    Features:
    - Metadata-aware embeddings (includes title, section_id in context)
    - Content change detection to avoid re-embedding unchanged documents
    - Persistent caching to disk to save API costs
    - Batch processing for efficient API usage
    """
    
    def __init__(
        self,
        api_key: Optional[Union[str, Secret]] = None,
        model_name: str = "text-embedding-3-small",
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_cache_dir: str = ".embedding_cache",
        progress_bar: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize the OpenAI document embedder.
        
        Args:
            api_key: OpenAI API key
            model_name: OpenAI embedding model name
            meta_fields_to_embed: Metadata fields to include in embedding context
            embedding_cache_dir: Directory for caching embeddings
            progress_bar: Whether to show progress during embedding
            batch_size: Number of documents to embed in each API call
        """
        self.api_key = Secret.from_token(api_key) if isinstance(api_key, str) else api_key
        self.model_name = model_name
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.progress_bar = progress_bar
        self.batch_size = batch_size
        
        # Create embedding cache
        self.cache = EmbeddingCache(cache_dir=embedding_cache_dir)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key.resolve_value() if self.api_key else None)
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Embed documents using OpenAI API.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Dictionary containing the embedded documents
        """
        if not documents:
            return {"documents": []}
        
        # Process documents in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i+self.batch_size]
            self._embed_batch(batch)
            
            if self.progress_bar and i % (self.batch_size * 5) == 0:
                logger.info(f"Embedded {i+len(batch)}/{len(documents)} documents")
        
        return {"documents": documents}
    
    def _embed_batch(self, documents: List[Document]) -> None:
        """
        Embed a batch of documents, using cache when possible.
        
        Args:
            documents: Batch of documents to embed
        """
        docs_to_embed = []
        texts_to_embed = []
        
        for i, doc in enumerate(documents):
            # Prepare document text with metadata
            text = self._prepare_text_for_embedding(doc)
            
            # Generate document ID and content hash
            doc_id = str(id(doc)) if not hasattr(doc, "id") or not doc.id else doc.id
            content_hash = self.cache._hash_document(doc)
            
            # Check cache first
            cached_embedding = self.cache.get_doc_embedding(doc_id, content_hash)
            
            if cached_embedding is not None:
                # Use cached embedding
                doc.embedding = cached_embedding
            else:
                # Need to embed this document
                docs_to_embed.append((i, doc, doc_id, content_hash))
                texts_to_embed.append(text)
        
        # If all documents were in cache, we're done
        if not texts_to_embed:
            return
        
        # Embed the texts that weren't cached
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts_to_embed
            )
            
            # Extract embeddings and update documents
            for i, (original_idx, doc, doc_id, content_hash) in enumerate(docs_to_embed):
                embedding = np.array(response.data[i].embedding)
                doc.embedding = embedding
                
                # Cache the embedding
                self.cache.save_doc_embedding(doc_id, content_hash, embedding)
                
        except Exception as e:
            logger.error(f"Error embedding documents with OpenAI: {e}")
            # For documents that failed embedding, set empty embedding
            for original_idx, doc, _, _ in docs_to_embed:
                if not hasattr(doc, "embedding") or doc.embedding is None:
                    # Use typical embedding dimensions for OpenAI models
                    dimensions = 1536 if "large" in self.model_name else 1024
                    doc.embedding = np.zeros(dimensions)
    
    def _prepare_text_for_embedding(self, doc: Document) -> str:
        """
        Combine document content and metadata for embedding.
        
        Args:
            doc: Document to prepare
            
        Returns:
            Text prepared for embedding
        """
        text = doc.content or ""
        
        # Include specified metadata fields
        if self.meta_fields_to_embed and doc.meta:
            meta_texts = []
            
            for field in self.meta_fields_to_embed:
                if field in doc.meta and doc.meta[field]:
                    meta_value = str(doc.meta[field])
                    meta_texts.append(f"{field}: {meta_value}")
            
            if meta_texts:
                text = text + "\n" + "\n".join(meta_texts)
        
        return text


@component
class OpenAITextEmbedder:
    """
    A Haystack component that embeds query texts using OpenAI's embedding models.
    
    Features:
    - Caches query embeddings to avoid redundant API calls
    - Compatible with InMemoryEmbeddingRetriever and other Haystack components
    - Handles API errors gracefully
    """
    
    def __init__(
        self,
        api_key: Optional[Union[str, Secret]] = None,
        model_name: str = "text-embedding-3-small",
        embedding_cache_dir: str = ".embedding_cache"
    ):
        """
        Initialize the OpenAI text embedder.
        
        Args:
            api_key: OpenAI API key
            model_name: OpenAI embedding model name
            embedding_cache_dir: Directory for caching embeddings
        """
        self.api_key = Secret.from_token(api_key) if isinstance(api_key, str) else api_key
        self.model_name = model_name
        
        # Create embedding cache
        self.cache = EmbeddingCache(cache_dir=embedding_cache_dir)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key.resolve_value() if self.api_key else None)
    
    @component.output_types(embedding=List[float])
    def run(self, text: str) -> Dict[str, List[float]]:
        """
        Embed text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            Dictionary containing the embedding
        """
        if not text:
            # Return zero vector for empty text
            dimensions = 1536 if "large" in self.model_name else 1024
            return {"embedding": [0.0] * dimensions}
        
        # Check cache first
        cached_embedding = self.cache.get_query_embedding(text)
        if cached_embedding is not None:
            return {"embedding": cached_embedding.tolist()}
        
        # Embed with OpenAI
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[text]
            )
            
            # Get embedding
            embedding = response.data[0].embedding
            
            # Cache the embedding
            self.cache.save_query_embedding(text, np.array(embedding))
            
            return {"embedding": embedding}
            
        except Exception as e:
            logger.error(f"Error embedding text with OpenAI: {e}")
            # Return zero vector on error
            dimensions = 1536 if "large" in self.model_name else 1024
            return {"embedding": [0.0] * dimensions}