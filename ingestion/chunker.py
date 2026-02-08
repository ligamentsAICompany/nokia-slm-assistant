"""
Chunker
=======
Fixed-size, deterministic text chunking for document ingestion.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import List, Iterator

from .pdf_loader import PageData

logger = logging.getLogger(__name__)


@dataclass
class ChunkData:
    """Data for a single text chunk."""
    chunk_id: str
    document_name: str
    page_number: int
    chunk_index: int  # Index within the document
    text: str
    char_count: int
    
    def to_metadata(self) -> dict:
        """Convert to metadata dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "page": self.page_number,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
        }


class Chunker:
    """
    Fixed-size, overlapping text chunker.
    
    Uses character-based chunking with deterministic IDs.
    
    Usage:
        chunker = Chunker(chunk_size=512, overlap=64)
        for chunk in chunker.chunk_pages(pages):
            print(chunk.chunk_id, chunk.text[:50])
    """
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        overlap: int = 64,
        min_chunk_size: int = 50
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target characters per chunk.
            overlap: Overlap characters between chunks.
            min_chunk_size: Minimum chunk size to emit.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
    
    def chunk_text(
        self, 
        text: str, 
        document_name: str,
        page_number: int,
        start_index: int = 0
    ) -> Iterator[ChunkData]:
        """
        Chunk a single text into fixed-size overlapping chunks.
        
        Args:
            text: Text to chunk.
            document_name: Source document name.
            page_number: Source page number.
            start_index: Starting chunk index (for multi-page docs).
            
        Yields:
            ChunkData for each chunk.
        """
        if not text or len(text) < self.min_chunk_size:
            return
        
        text = text.strip()
        text_len = len(text)
        
        step = self.chunk_size - self.overlap
        chunk_index = start_index
        
        for start in range(0, text_len, step):
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) < self.min_chunk_size:
                continue
            
            # Generate deterministic chunk ID
            chunk_id = self._generate_chunk_id(
                document_name, page_number, chunk_index, chunk_text
            )
            
            yield ChunkData(
                chunk_id=chunk_id,
                document_name=document_name,
                page_number=page_number,
                chunk_index=chunk_index,
                text=chunk_text,
                char_count=len(chunk_text)
            )
            
            chunk_index += 1
            
            # Stop if we've reached the end
            if end >= text_len:
                break
    
    def chunk_page(self, page: PageData, start_index: int = 0) -> List[ChunkData]:
        """
        Chunk a single page.
        
        Args:
            page: PageData to chunk.
            start_index: Starting chunk index.
            
        Returns:
            List of ChunkData.
        """
        return list(self.chunk_text(
            page.text,
            page.document_name,
            page.page_number,
            start_index
        ))
    
    def chunk_pages(self, pages: Iterator[PageData]) -> Iterator[ChunkData]:
        """
        Chunk multiple pages, maintaining continuous chunk indices.
        
        Args:
            pages: Iterator of PageData.
            
        Yields:
            ChunkData for each chunk.
        """
        global_index = 0
        
        for page in pages:
            for chunk in self.chunk_text(
                page.text,
                page.document_name,
                page.page_number,
                global_index
            ):
                yield chunk
                global_index += 1
    
    def _generate_chunk_id(
        self, 
        doc_name: str, 
        page: int, 
        index: int, 
        text: str
    ) -> str:
        """
        Generate a deterministic chunk ID.
        
        Uses hash of content + metadata for reproducibility.
        """
        # Include first 100 chars of text for uniqueness
        content = f"{doc_name}:{page}:{index}:{text[:100]}"
        hash_hex = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"{doc_name}_p{page}_c{index}_{hash_hex}"
    
    @staticmethod
    def estimate_chunk_count(text_length: int, chunk_size: int, overlap: int) -> int:
        """Estimate number of chunks for given text length."""
        if text_length <= chunk_size:
            return 1
        step = chunk_size - overlap
        return ((text_length - chunk_size) // step) + 1
