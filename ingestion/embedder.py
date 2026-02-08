"""
Embedder
========
FastEmbed wrapper for generating document embeddings.
"""

import logging
from typing import List, Iterator, Optional
import numpy as np

from .chunker import ChunkData

logger = logging.getLogger(__name__)


class Embedder:
    """
    FastEmbed-based text embedder.
    
    Uses BAAI/bge-small-en-v1.5 (384 dimensions) for efficient CPU embedding.
    
    Usage:
        embedder = Embedder()
        embeddings = embedder.embed_texts(["Hello world", "Goodbye world"])
    """
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: FastEmbed model name.
            batch_size: Batch size for embedding.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._dimension = None
    
    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                logger.info(f"Loading FastEmbed model: {self.model_name}")
                self._model = TextEmbedding(model_name=self.model_name)
                
                # Determine dimension by embedding a test string
                test_emb = list(self._model.embed(["test"]))[0]
                self._dimension = len(test_emb)
                logger.info(f"Model loaded. Dimension: {self._dimension}")
            except Exception as e:
                logger.error(f"Failed to load FastEmbed: {e}")
                raise
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self._load_model()
        return self._dimension
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as numpy array.
        """
        self._load_model()
        embeddings = list(self._model.embed([text]))
        return np.array(embeddings[0], dtype=np.float32)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            Numpy array of shape (len(texts), dimension).
        """
        self._load_model()
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = list(self._model.embed(batch))
            all_embeddings.extend(batch_embeddings)
            
            if (i + self.batch_size) % 100 == 0:
                logger.debug(f"Embedded {i + len(batch)}/{len(texts)} texts")
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def embed_chunks(
        self, 
        chunks: List[ChunkData],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed a list of chunks.
        
        Args:
            chunks: List of ChunkData to embed.
            show_progress: Whether to log progress.
            
        Returns:
            Numpy array of embeddings.
        """
        texts = [chunk.text for chunk in chunks]
        
        if show_progress:
            logger.info(f"Embedding {len(texts)} chunks...")
        
        embeddings = self.embed_texts(texts)
        
        if show_progress:
            logger.info(f"Embedding complete. Shape: {embeddings.shape}")
        
        return embeddings
    
    def embed_chunks_streaming(
        self, 
        chunks: Iterator[ChunkData],
        buffer_size: int = 1000
    ) -> Iterator[tuple]:
        """
        Embed chunks in a streaming fashion.
        
        Args:
            chunks: Iterator of ChunkData.
            buffer_size: Number of chunks to buffer before embedding.
            
        Yields:
            Tuple of (ChunkData, embedding_vector).
        """
        self._load_model()
        
        buffer = []
        
        for chunk in chunks:
            buffer.append(chunk)
            
            if len(buffer) >= buffer_size:
                embeddings = self.embed_chunks(buffer, show_progress=False)
                for i, emb in enumerate(embeddings):
                    yield buffer[i], emb
                buffer = []
        
        # Process remaining
        if buffer:
            embeddings = self.embed_chunks(buffer, show_progress=False)
            for i, emb in enumerate(embeddings):
                yield buffer[i], emb
