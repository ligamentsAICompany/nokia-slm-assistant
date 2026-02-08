"""
FAISS Indexer
=============
Builds and manages FAISS vector index and metadata store.
"""

import logging
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from .chunker import ChunkData

logger = logging.getLogger(__name__)


class FAISSIndexer:
    """
    FAISS index builder and manager.
    
    Creates a flat L2 index for vector similarity search.
    Also manages the metadata store (pickle format).
    
    Usage:
        indexer = FAISSIndexer(dimension=384)
        indexer.add_chunks(chunks, embeddings)
        indexer.save("path/to/index.faiss", "path/to/meta.pkl")
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize indexer.
        
        Args:
            dimension: Embedding dimension.
        """
        self.dimension = dimension
        self._index = None
        self._metadata: List[Dict] = []
    
    def _create_index(self):
        """Create a new FAISS index."""
        import faiss
        self._index = faiss.IndexFlatL2(self.dimension)
        logger.info(f"Created FAISS IndexFlatL2 with dimension {self.dimension}")
    
    def add_vectors(
        self, 
        embeddings: np.ndarray,
        metadata_list: List[Dict]
    ):
        """
        Add vectors and metadata to the index.
        
        Args:
            embeddings: Numpy array of shape (n, dimension).
            metadata_list: List of metadata dicts for each vector.
        """
        if self._index is None:
            self._create_index()
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.dimension}"
            )
        
        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"Count mismatch: {len(embeddings)} embeddings, {len(metadata_list)} metadata"
            )
        
        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        
        # Add to index
        self._index.add(embeddings)
        self._metadata.extend(metadata_list)
        
        logger.info(f"Added {len(embeddings)} vectors. Total: {self._index.ntotal}")
    
    def add_chunks(
        self, 
        chunks: List[ChunkData],
        embeddings: np.ndarray
    ):
        """
        Add chunks with their embeddings.
        
        Args:
            chunks: List of ChunkData.
            embeddings: Corresponding embeddings.
        """
        metadata_list = []
        for chunk in chunks:
            meta = chunk.to_metadata()
            meta['original_text'] = chunk.text
            meta['enriched_text'] = chunk.text  # Can be enhanced later
            metadata_list.append(meta)
        
        self.add_vectors(embeddings, metadata_list)
    
    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        if self._index is None:
            return 0
        return self._index.ntotal
    
    def save(
        self, 
        index_path: str, 
        metadata_path: str,
        backup: bool = True
    ):
        """
        Save index and metadata to disk.
        
        Args:
            index_path: Path for FAISS index file.
            metadata_path: Path for metadata pickle file.
            backup: Whether to backup existing files.
        """
        import faiss
        
        if self._index is None:
            raise ValueError("No index to save")
        
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        
        # Create directories
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing files
        if backup:
            if index_path.exists():
                backup_path = index_path.with_suffix('.faiss.backup')
                shutil.copy2(index_path, backup_path)
                logger.info(f"Backed up existing index to {backup_path}")
            
            if metadata_path.exists():
                backup_path = metadata_path.with_suffix('.pkl.backup')
                shutil.copy2(metadata_path, backup_path)
                logger.info(f"Backed up existing metadata to {backup_path}")
        
        # Write atomically (write to temp, then rename)
        temp_index = index_path.with_suffix('.faiss.tmp')
        temp_meta = metadata_path.with_suffix('.pkl.tmp')
        
        try:
            faiss.write_index(self._index, str(temp_index))
            with open(temp_meta, 'wb') as f:
                pickle.dump(self._metadata, f)
            
            # Atomic rename
            temp_index.rename(index_path)
            temp_meta.rename(metadata_path)
            
            logger.info(f"Saved FAISS index ({self._index.ntotal} vectors) to {index_path}")
            logger.info(f"Saved metadata ({len(self._metadata)} entries) to {metadata_path}")
            
        except Exception as e:
            # Cleanup temp files on failure
            temp_index.unlink(missing_ok=True)
            temp_meta.unlink(missing_ok=True)
            raise
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> 'FAISSIndexer':
        """
        Load index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file.
            metadata_path: Path to metadata pickle file.
            
        Returns:
            FAISSIndexer instance.
        """
        import faiss
        
        index = faiss.read_index(str(index_path))
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        indexer = cls(dimension=index.d)
        indexer._index = index
        indexer._metadata = metadata
        
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        
        return indexer
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10
    ) -> List[Dict]:
        """
        Search the index.
        
        Args:
            query_embedding: Query vector.
            k: Number of results.
            
        Returns:
            List of metadata dicts with scores.
        """
        if self._index is None:
            return []
        
        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._metadata):
                meta = self._metadata[idx].copy()
                meta['score'] = float(dist)
                results.append(meta)
        
        return results
