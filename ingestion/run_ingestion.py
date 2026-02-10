"""
Nokia SLM Ingestion Pipeline
============================
Main entry point for running the full ingestion pipeline.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from .pdf_loader import PDFLoader, load_all_pdfs
from .chunker import Chunker, ChunkData
from .embedder import Embedder
from .indexer import FAISSIndexer
from .graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


def run_full_ingestion(
    documents_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    build_graph: bool = True,
    force: bool = False
) -> dict:
    """
    Run the complete ingestion pipeline.
    
    Args:
        documents_dir: Directory containing PDF documents.
        output_dir: Directory for output files (index, metadata, graph).
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.
        build_graph: Whether to build knowledge graph.
        force: Overwrite existing files without prompting.
        
    Returns:
        Dictionary with ingestion statistics.
    """
    config = load_config()
    
    # Resolve directories
    docs_dir = Path(documents_dir or config.documents_dir)
    out_dir = Path(output_dir or config.data_dir)
    
    # Ensure directories exist
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Output paths
    index_path = out_dir / "nokia_vector_index.faiss"
    meta_path = out_dir / "nokia_vector_meta.pkl"
    
    # Check for existing files
    if not force:
        existing = []
        if index_path.exists():
            existing.append(str(index_path))
        if meta_path.exists():
            existing.append(str(meta_path))
        if existing:
            logger.warning(f"Existing files will be backed up: {existing}")
    
    stats = {
        "documents": 0,
        "pages": 0,
        "chunks": 0,
        "vectors": 0,
        "graph_nodes": 0,
        "graph_edges": 0,
        "duration_seconds": 0,
    }
    
    start_time = time.time()
    
    # Step 1: Find PDFs
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {docs_dir}")
        return stats
    
    stats["documents"] = len(pdf_files)
    logger.info(f"Found {len(pdf_files)} PDF documents")
    
    # Step 2: Initialize components
    chunker = Chunker(chunk_size=chunk_size, overlap=chunk_overlap)
    embedder = Embedder(model_name=config.embedding_model)
    indexer = FAISSIndexer(dimension=config.embedding_dim)
    
    if build_graph:
        graph_builder = GraphBuilder()
    
    # Step 3: Process each PDF
    all_chunks = []
    
    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        
        with PDFLoader(str(pdf_path)) as loader:
            pages = list(loader.load_pages())
            stats["pages"] += len(pages)
            
            # Chunk all pages
            for page in pages:
                page_chunks = chunker.chunk_page(page, start_index=len(all_chunks))
                all_chunks.extend(page_chunks)
    
    stats["chunks"] = len(all_chunks)
    logger.info(f"Total chunks: {len(all_chunks)}")
    
    if not all_chunks:
        logger.error("No chunks generated. Check PDF content.")
        return stats
    
    # Step 4: Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embedder.embed_chunks(all_chunks, show_progress=True)
    
    # Step 5: Build FAISS index
    logger.info("Building FAISS index...")
    indexer.add_chunks(all_chunks, embeddings)
    indexer.save(str(index_path), str(meta_path))
    
    stats["vectors"] = indexer.size
    
    # Step 6: Build knowledge graph in Neo4j (optional)
    if build_graph:
        logger.info("Building knowledge graph in Neo4j...")
        graph_builder.process_chunks(all_chunks, doc_id="nokia_ingestion", show_progress=True)
        graph_builder.add_predefined_relationships()
        graph_builder.save()  # no-op — Neo4j persists automatically
        
        graph_stats = graph_builder.get_stats()
        stats["graph_nodes"] = graph_stats.get("total_nodes", graph_stats.get("nodes", 0))
        stats["graph_edges"] = graph_stats.get("total_relationships", graph_stats.get("edges", 0))
    
    stats["duration_seconds"] = round(time.time() - start_time, 2)
    
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Documents:    {stats['documents']}")
    logger.info(f"Pages:        {stats['pages']}")
    logger.info(f"Chunks:       {stats['chunks']}")
    logger.info(f"Vectors:      {stats['vectors']}")
    if build_graph:
        logger.info(f"Graph Nodes:  {stats['graph_nodes']}")
        logger.info(f"Graph Edges:  {stats['graph_edges']}")
    logger.info(f"Duration:     {stats['duration_seconds']}s")
    logger.info("=" * 60)
    
    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Nokia SLM Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ingestion.run_ingestion
  python -m ingestion.run_ingestion --documents ./pdfs --output ./data
  python -m ingestion.run_ingestion --chunk-size 1024 --no-graph
        """
    )
    
    parser.add_argument(
        "--documents", "-d",
        type=str,
        default=None,
        help="Directory containing PDF documents"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for index and metadata"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Characters per chunk (default: 512)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Overlap between chunks (default: 64)"
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip knowledge graph generation"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing files without warning"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    
    # Run ingestion
    try:
        stats = run_full_ingestion(
            documents_dir=args.documents,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            build_graph=not args.no_graph,
            force=args.force
        )
        
        if stats["vectors"] > 0:
            print("\nIngestion successful!")
            sys.exit(0)
        else:
            print("\nIngestion completed with warnings. Check logs.")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
