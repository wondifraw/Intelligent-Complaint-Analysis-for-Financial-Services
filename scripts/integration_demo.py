"""
Integration Demo: Complete Workflow from Text Chunking to Embedding and Indexing

This script demonstrates the complete pipeline:
1. Load complaint data
2. Chunk narratives using the text chunking module
3. Generate embeddings using sentence-transformers/all-MiniLM-L6-v2
4. Index in ChromaDB with full metadata preservation
5. Perform semantic search with traceability
"""

import sys
from pathlib import Path
import pandas as pd
import time

sys.path.append(str(Path(__file__).parent.parent / "src"))

from text_chunking import NarrativeChunkingStrategy
from embedding_indexing import (
    ComplaintEmbeddingIndexer,
    create_embedding_indexer,
    search_complaints_by_topic
)


def load_complaints_data():
    """Load financial complaint data from CSV file using chunked reading."""
    data_path = Path(__file__).parent.parent / "data" / "complaints_processed.csv"
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return None, None
    
    try:
        # Load the CSV file using chunked reading to handle memory issues
        chunk_size = 100000  # Adjust this number based on your available memory
        print(f"Loading data in chunks of {chunk_size} rows...")
        
        chunks = []
        for chunk in pd.read_csv(data_path, low_memory=False, chunksize=chunk_size):
            # Optionally process each chunk here (e.g., filter, clean, etc.)
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        
        # For demo purposes, limit to first 100 complaints
        df = df.head(100)
        
        # Find narrative column
        narrative_column = 'Consumer complaint narrative'
        if narrative_column not in df.columns:
            for col in ['Consumer complaint narrative_cleaned', 'narrative', 'text']:
                if col in df.columns:
                    narrative_column = col
                    break
            else:
                print(f"Could not find narrative column. Available: {list(df.columns)}")
                return None, None
        
        # Filter out empty narratives
        df = df.dropna(subset=[narrative_column])
        df = df[df[narrative_column].str.strip() != '']
        
        print(f"Loaded {len(df)} complaints")
        return df, narrative_column
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def chunk_narratives(df, narrative_column):
    """Chunk narratives using the text chunking module."""
    print("\n=== Step 2: Text Chunking ===")
    
    strategy = NarrativeChunkingStrategy(
        chunk_size=200,
        chunk_overlap=50,
        chunking_method="recursive_character"
    )
    
    narratives = df[narrative_column].tolist()
    print(f"Chunking {len(narratives)} narratives...")
    
    start_time = time.time()
    chunked_narratives = strategy.chunk_narratives_batch(narratives)
    chunking_time = time.time() - start_time
    
    print(f"Created {len(chunked_narratives)} chunks in {chunking_time:.2f}s")
    
    # Add original metadata to chunks
    for chunk in chunked_narratives:
        original_idx = chunk['original_narrative_index']
        if original_idx < len(df):
            chunk['original_Product'] = df.iloc[original_idx].get('Product', 'Unknown')
            chunk['original_Company'] = df.iloc[original_idx].get('Company', 'Unknown')
    
    return chunked_narratives


def create_embeddings_and_index(chunked_narratives):
    """Generate embeddings and index in FAISS."""
    print("\n=== Step 3: Embedding Generation and Indexing ===")
    
    print("Initializing embedding model: sentence-transformers/all-MiniLM-L6-v2")
    indexer = create_embedding_indexer()
    
    print(f"Indexing {len(chunked_narratives)} chunks...")
    start_time = time.time()
    
    success = indexer.index_chunks(chunked_narratives)
    
    if success:
        indexing_time = time.time() - start_time
        print(f"Successfully indexed chunks in {indexing_time:.2f}s")
        
        stats = indexer.get_collection_stats()
        print(f"Collection: {stats['total_chunks']} chunks")
        # Save FAISS index and metadata
        save_success = indexer.save_faiss_index("vector_store")
        if save_success:
            print("FAISS index and metadata saved to vector_store/")
        else:
            print("Failed to save FAISS index and metadata.")
        return indexer
    else:
        print("Failed to index chunks")
        return None


def demonstrate_search_capabilities(indexer):
    """Demonstrate semantic search capabilities."""
    print("\n=== Step 4: Semantic Search Demonstration ===")
    
    test_queries = [
        "fraudulent charges on credit card",
        "poor customer service",
        "unauthorized transactions",
        "mortgage payment issues"
    ]
    
    for query in test_queries:
        print(f"\n🔍 '{query}':")
        
        results = indexer.search_similar_chunks(query, n_results=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['text'][:80]}...")
                print(f"     Score: {1 - result['distance']:.3f}")
                print(f"     Product: {result['metadata'].get('original_Product', 'N/A')}")
        else:
            print("  No results found")


def demonstrate_filtered_search(indexer):
    """Demonstrate filtered search capabilities."""
    print("\n=== Step 5: Filtered Search ===")
    
    print("🔍 Searching for 'fraud' in Credit card products only:")
    
    results = search_complaints_by_topic(
        query="fraud",
        indexer=indexer,
        n_results=2,
        product_category="Credit card"
    )
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['text'][:80]}...")
            print(f"     Product: {result['metadata'].get('original_Product', 'N/A')}")
    else:
        print("  No results found")


def demonstrate_traceability(indexer):
    """Demonstrate traceability features."""
    print("\n=== Step 6: Traceability ===")
    
    results = indexer.search_similar_chunks("customer service issues", n_results=1)
    
    if results:
        result = results[0]
        print(f"📋 Traceability Example:")
        print(f"Search Result: {result['text'][:80]}...")
        print(f"\n📊 Traceability Information:")
        print(f"  Original Index: {result['metadata'].get('original_narrative_index', 'N/A')}")
        print(f"  Chunk Index: {result['metadata'].get('chunk_index', 'N/A')}")
        print(f"  Product: {result['metadata'].get('original_Product', 'N/A')}")
        print(f"  Company: {result['metadata'].get('original_Company', 'N/A')}")
        print(f"  Similarity Score: {1 - result['distance']:.3f}")


def export_results(indexer):
    """Export results for analysis."""
    print("\n=== Step 7: Export Results ===")
    
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / "demo_embeddings.json"
    success = indexer.export_collection_data(str(json_path), "json")
    
    if success:
        print(f"✅ Exported embeddings to: {json_path}")
    else:
        print("❌ Failed to export embeddings")


def main():
    """Run the complete integration demo."""
    print("🚀 Financial Complaint Analysis - Complete Integration Demo")
    print("=" * 60)
    print("This demo shows the complete workflow from text chunking to")
    print("embedding generation and semantic search with full traceability.")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        print("\n=== Step 1: Data Loading ===")
        df, narrative_column = load_complaints_data()
        
        if df is None:
            print("❌ No data available. Demo cannot continue.")
            return
        
        # Step 2: Chunk narratives
        chunked_narratives = chunk_narratives(df, narrative_column)
        
        # Step 3: Create embeddings and index
        indexer = create_embeddings_and_index(chunked_narratives)
        
        if indexer is None:
            print("❌ Failed to create embeddings and index. Demo cannot continue.")
            return
        
        # Step 4: Demonstrate search capabilities
        demonstrate_search_capabilities(indexer)
        
        # Step 5: Demonstrate filtered search
        demonstrate_filtered_search(indexer)
        
        # Step 6: Demonstrate traceability
        demonstrate_traceability(indexer)
        
        # Step 7: Export results
        export_results(indexer)
        
        print("\n" + "=" * 60)
        print("✅ Integration Demo Completed Successfully!")
        print(f"\n📋 Summary:")
        print(f"   • Processed {len(df)} complaints")
        print(f"   • Created {len(chunked_narratives)} text chunks")
        print(f"   • Generated embeddings using sentence-transformers/all-MiniLM-L6-v2")
        print(f"   • Indexed in FAISS with full metadata preservation")
        print(f"   • Demonstrated semantic search with traceability")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")


if __name__ == "__main__":
    main() 