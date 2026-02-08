"""
PDF Loader
==========
Loads Nokia PDF documents using PyMuPDF.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    """Data extracted from a single PDF page."""
    document_name: str
    page_number: int
    text: str
    char_count: int
    
    def __repr__(self) -> str:
        return f"PageData(doc={self.document_name}, page={self.page_number}, chars={self.char_count})"


class PDFLoader:
    """
    PDF document loader using PyMuPDF (fitz).
    
    Usage:
        loader = PDFLoader("documents/nokia_manual.pdf")
        for page in loader.load_pages():
            print(page.page_number, page.text[:100])
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF loader.
        
        Args:
            pdf_path: Path to PDF file.
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.document_name = self.pdf_path.stem
        self._doc = None
    
    def open(self):
        """Open the PDF document."""
        import fitz  # PyMuPDF
        self._doc = fitz.open(str(self.pdf_path))
        logger.info(f"Opened PDF: {self.document_name} ({self._doc.page_count} pages)")
    
    def close(self):
        """Close the PDF document."""
        if self._doc:
            self._doc.close()
            self._doc = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    @property
    def page_count(self) -> int:
        """Get total page count."""
        if self._doc is None:
            self.open()
        return self._doc.page_count
    
    def load_page(self, page_num: int) -> PageData:
        """
        Load a single page by number (0-indexed).
        
        Args:
            page_num: Page number (0-indexed).
            
        Returns:
            PageData for the page.
        """
        if self._doc is None:
            self.open()
        
        page = self._doc[page_num]
        text = page.get_text("text")
        
        # Clean up text
        text = self._clean_text(text)
        
        return PageData(
            document_name=self.document_name,
            page_number=page_num + 1,  # 1-indexed for display
            text=text,
            char_count=len(text)
        )
    
    def load_pages(self) -> Iterator[PageData]:
        """
        Iterate over all pages in the document.
        
        Yields:
            PageData for each page.
        """
        if self._doc is None:
            self.open()
        
        for page_num in range(self._doc.page_count):
            yield self.load_page(page_num)
    
    def load_all_pages(self) -> List[PageData]:
        """
        Load all pages into a list.
        
        Returns:
            List of PageData for all pages.
        """
        return list(self.load_pages())
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw extracted text.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Replace multiple whitespace with single space
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers (common patterns)
        # This is document-specific; adjust as needed
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()


def load_all_pdfs(documents_dir: str) -> Iterator[PageData]:
    """
    Load all PDFs from a directory.
    
    Args:
        documents_dir: Path to directory containing PDFs.
        
    Yields:
        PageData for each page of each PDF.
    """
    docs_path = Path(documents_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    
    pdf_files = list(docs_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {documents_dir}")
    
    for pdf_path in sorted(pdf_files):
        logger.info(f"Processing: {pdf_path.name}")
        with PDFLoader(str(pdf_path)) as loader:
            for page in loader.load_pages():
                yield page
