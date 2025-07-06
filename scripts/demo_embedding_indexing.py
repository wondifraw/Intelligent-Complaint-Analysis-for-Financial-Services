"""
Demo script for embedding and indexing module.

This script demonstrates the embedding and indexing capabilities using
sentence-transformers/all-MiniLM-L6-v2 and ChromaDB for financial complaint analysis.
"""

import sys
from pathlib import Path
import pandas as pd
import time
import gc
import os
import psutil

sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import with error handling
try:
    from embedding_indexing import (
        ComplaintEmbeddingIndexer,
        create_embedding_indexer,
        batch_index_from_csv,
        search_complaints_by_topic,
        validate_chunk_data
    )
    EMBEDDING_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing embedding_indexing module: {e}")
    EMBEDDING_MODULE_AVAILABLE = False


def log_memory_usage(context=""):
    try:
        memory = psutil.virtual_memory()
        print(f"[MEMORY] {context} | Available: {memory.available / (1024**3):.2f} GB | Used: {memory.percent}%")
    except Exception:
        pass


def safe_load_chunked_data(max_chunks=1000):
    """Load chunked narratives from CSV file with safety limits and memory logging."""
    log_memory_usage("Before loading chunked data")
    if not EMBEDDING_MODULE_AVAILABLE:
        print("❌ Embedding module not available")
        return None
    
    data_path = Path(__file__).parent.parent / "data" / "chunked_narratives.csv"
    
    if not data_path.exists():
        print(f"❌ Chunked data file not found: {data_path}")
        return None
    
    try:
        # Check file size first
        file_size = data_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            print(f"⚠️  Large file detected ({file_size / 1024 / 1024:.1f}MB). Loading limited sample.")
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                # Load with chunk limit
                df = pd.read_csv(data_path, encoding=encoding, nrows=max_chunks)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"❌ Error reading CSV with {encoding}: {e}")
                continue
        else:
            print("❌ Failed to read CSV file")
            return None
        
        if df.empty or 'chunk_text' not in df.columns:
            print("❌ Invalid CSV file")
            return None
        
        # Filter out empty chunks
        df = df.dropna(subset=['chunk_text'])
        df = df[df['chunk_text'].str.strip() != '']
        
        print(f"✅ Loaded {len(df)} chunks (limited to {max_chunks})")
        log_memory_usage("After loading chunked data")
        return df.to_dict('records')
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    finally:
        gc.collect()
        log_memory_usage("After gc.collect() in safe_load_chunked_data")


def load_chunked_data(max_chunks=1000):
    """Load chunked data with a configurable chunk limit for memory safety."""
    return safe_load_chunked_data(max_chunks=max_chunks)


def lightweight_demo():
    """Lightweight demo for low-memory situations."""
    print("🚀 Lightweight Embedding Demo (Low Memory Mode)")
    print("=" * 50)
    
    if not EMBEDDING_MODULE_AVAILABLE:
        print("❌ Embedding module not available")
        return
    
    try:
        # Create indexer with minimal settings
        print("🔄 Creating lightweight indexer...")
        indexer = create_embedding_indexer(
            chroma_persist_directory="./lightweight_chroma_db",
            collection_name="lightweight_demo"
        )
        
        # Test with very small data
        test_texts = [
            "I have a complaint about my credit card.",
            "The bank refused to help me.",
            "My account was charged incorrectly."
        ]
        
        print("🔄 Generating embeddings for 3 test texts...")
        embeddings = indexer.generate_embeddings(test_texts)
        
        if embeddings:
            print(f"✅ Generated {len(embeddings)} embeddings")
            print(f"📏 Embedding dimension: {len(embeddings[0])}")
            
            # Simple search test
            print("\n🔍 Testing search functionality...")
            results = indexer.search_similar_chunks("credit card complaint", n_results=1)
            
            if results:
                print(f"✅ Search successful: {results[0]['text'][:50]}...")
            else:
                print("⚠️  No search results (expected with small dataset)")
        else:
            print("❌ Failed to generate embeddings")
            
    except Exception as e:
        print(f"❌ Error in lightweight demo: {e}")
    finally:
        gc.collect()
    
    print("\n" + "=" * 50)
    print("🏁 Lightweight demo completed!")


def safe_create_indexer():
    """Safely create embedding indexer with error handling."""
    if not EMBEDDING_MODULE_AVAILABLE:
        return None
    
    try:
        # Check available memory
        memory = psutil.virtual_memory()
        if memory.available < 2 * 1024 * 1024 * 1024:  # 2GB
            print("⚠️  Low memory available. Consider closing other applications.")
            print("💡 Try running lightweight_demo() instead")
            return None
    except ImportError:
        pass  # psutil not available
    
    try:
        indexer = create_embedding_indexer()
        print("✅ Embedding indexer created successfully")
        return indexer
    except Exception as e:
        print(f"❌ Error creating indexer: {e}")
        return None


def demo_embedding_model():
    """Demonstrate embedding model choice and basic functionality with memory logging."""
    print("=== Embedding Model: sentence-transformers/all-MiniLM-L6-v2 ===")
    print("• Lightweight and fast")
    print("• 384-dimensional embeddings")
    print("• Excellent semantic similarity performance")
    indexer = safe_create_indexer()
    if not indexer:
        return
    try:
        test_texts = [
            "I have a complaint about my credit card being charged incorrectly.",
            "The bank refused to refund unauthorized charges on my account.",
            "My credit limit was reduced without any prior notice."
        ]
        log_memory_usage("Before generating embeddings")
        print("🔄 Generating embeddings...")
        embeddings = indexer.generate_embeddings(test_texts)
        if embeddings:
            print(f"✅ Generated {len(embeddings)} embeddings")
            print(f"📏 Embedding dimension: {len(embeddings[0])}")
        else:
            print("❌ Failed to generate embeddings")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        gc.collect()
        log_memory_usage("After gc.collect() in demo_embedding_model")


def demo_indexing():
    """Demonstrate indexing chunked narratives with memory logging."""
    print("\n=== Indexing Chunked Narratives ===")
    log_memory_usage("Before loading chunks for indexing")
    chunks = safe_load_chunked_data(max_chunks=20)  # Reduced limit for stability
    if not chunks:
        return
    
    indexer = safe_create_indexer()
    if not indexer:
        return
    
    try:
        print(f"📊 Using {len(chunks)} chunks for demo")
        start_time = time.time()
        batch_size = 5
        total_processed = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            try:
                log_memory_usage(f"Before indexing batch {i//batch_size + 1}")
                success = indexer.index_chunks(batch)
                if success:
                    total_processed += len(batch)
                    print(f"✅ Batch {i//batch_size + 1} completed")
                else:
                    print(f"⚠️  Batch {i//batch_size + 1} failed")
                gc.collect()
                log_memory_usage(f"After gc.collect() batch {i//batch_size + 1}")
            except Exception as batch_error:
                print(f"❌ Batch {i//batch_size + 1} error: {batch_error}")
                continue
        indexing_time = time.time() - start_time
        print(f"✅ Indexed {total_processed}/{len(chunks)} chunks in {indexing_time:.2f}s")
        if total_processed > 0:
            stats = indexer.get_collection_stats()
            print(f"📊 Collection: {stats['total_chunks']} chunks")
        else:
            print("❌ No chunks were successfully indexed")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        gc.collect()
        log_memory_usage("After gc.collect() in demo_indexing")


def demo_search():
    """Demonstrate semantic search functionality."""
    print("\n=== Semantic Search ===")
    
    indexer = safe_create_indexer()
    if not indexer:
        return
    
    try:
        test_queries = [
            "fraudulent charges on my credit card",
            "unauthorized transactions",
            "credit limit reduction"
        ]
        
        for query in test_queries:
            print(f"\n🔍 '{query}':")
            results = indexer.search_similar_chunks(query, n_results=2)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['text'][:80]}...")
                    print(f"     Score: {1 - result['distance']:.3f}")
            else:
                print("  No results found")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        gc.collect()


def demo_search_functionality():
    """Demonstrate comprehensive search functionality."""
    print("\n=== Comprehensive Search Functionality ===")
    
    indexer = safe_create_indexer()
    if not indexer:
        return
    
    try:
        # Test different types of queries
        search_scenarios = [
            {
                "name": "Fraud Detection",
                "query": "fraudulent charges on my credit card",
                "description": "Searching for fraud-related complaints"
            },
            {
                "name": "Customer Service",
                "query": "poor customer service experience",
                "description": "Searching for customer service issues"
            },
            {
                "name": "Unauthorized Transactions",
                "query": "unauthorized transactions on my account",
                "description": "Searching for unauthorized activity"
            },
            {
                "name": "Credit Issues",
                "query": "credit limit reduction without notice",
                "description": "Searching for credit-related problems"
            }
        ]
        
        for scenario in search_scenarios:
            print(f"\n🔍 {scenario['name']}:")
            print(f"   Query: '{scenario['query']}'")
            print(f"   Description: {scenario['description']}")
            
            results = indexer.search_similar_chunks(scenario['query'], n_results=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"   Result {i}:")
                    print(f"     Text: {result['text'][:100]}...")
                    print(f"     Similarity Score: {1 - result['distance']:.4f}")
                    if result['metadata']:
                        print(f"     Product: {result['metadata'].get('original_Product', 'N/A')}")
                        print(f"     Company: {result['metadata'].get('original_Company', 'N/A')}")
            else:
                print("   No results found")
        
        # Demonstrate filtered search
        print(f"\n🔍 Filtered Search - Credit Card Fraud Only:")
        filtered_results = search_complaints_by_topic(
            query="fraud",
            indexer=indexer,
            n_results=2,
            product_category="Credit card"
        )
        
        if filtered_results:
            for i, result in enumerate(filtered_results, 1):
                print(f"   Filtered Result {i}:")
                print(f"     Text: {result['text'][:80]}...")
                print(f"     Product: {result['metadata'].get('original_Product', 'N/A')}")
        else:
            print("   No filtered results found")
            
    except Exception as e:
        print(f"❌ Error in search functionality demo: {e}")
    finally:
        gc.collect()


def demo_metadata():
    """Demonstrate metadata preservation."""
    print("\n=== Metadata Preservation ===")
    
    indexer = safe_create_indexer()
    if not indexer:
        return
    
    try:
        # Create sample chunks with metadata
        sample_chunks = [
            {
                'chunk_text': 'I have a complaint about unauthorized charges on my credit card.',
                'original_narrative_index': 0,
                'chunk_index': 0,
                'original_Product': 'Credit card',
                'original_Company': 'Bank of America'
            }
        ]
        
        if indexer.index_chunks(sample_chunks):
            results = indexer.search_similar_chunks("fraudulent charges", n_results=1)
            if results:
                result = results[0]
                print(f"✅ Found: {result['text']}")
                print(f"📋 Metadata: {result['metadata']}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        gc.collect()


def demo_batch_processing():
    """Demonstrate batch processing from CSV."""
    print("\n=== Batch Processing ===")
    
    csv_path = Path(__file__).parent.parent / "data" / "chunked_narratives.csv"
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    indexer = safe_create_indexer()
    if not indexer:
        return
    
    try:
        start_time = time.time()
        
        # Use smaller batch size for demo
        success = batch_index_from_csv(str(csv_path), indexer, batch_size=50)
        
        if success:
            processing_time = time.time() - start_time
            print(f"✅ Batch processing completed in {processing_time:.2f}s")
            
            stats = indexer.get_collection_stats()
            print(f"📊 Total chunks: {stats['total_chunks']}")
        else:
            print("❌ Batch processing failed")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        gc.collect()


def demo_topic_search():
    """Demonstrate topic-based search with filtering."""
    print("\n=== Topic-Based Search ===")
    
    indexer = safe_create_indexer()
    if not indexer:
        return
    
    try:
        results = search_complaints_by_topic(
            query="fraud",
            indexer=indexer,
            n_results=2,
            product_category="Credit card"
        )
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['text'][:80]}...")
                print(f"   Product: {result['metadata'].get('original_Product', 'N/A')}")
        else:
            print("No results found")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        gc.collect()


def demo_validation():
    """Demonstrate data validation."""
    print("\n=== Data Validation ===")
    
    chunks = safe_load_chunked_data(max_chunks=100)  # Limit for validation
    if not chunks:
        return
    
    try:
        validation_results = validate_chunk_data(chunks)
        
        print(f"📊 Validation Results:")
        print(f"  Total: {validation_results['total_chunks']}")
        print(f"  Valid: {validation_results['valid_chunks']}")
        print(f"  Invalid: {validation_results['invalid_chunks']}")
        print(f"  Missing text: {validation_results['missing_text']}")
        print(f"  Empty text: {validation_results['empty_text']}")
        
        quality_rate = validation_results['valid_chunks'] / validation_results['total_chunks']
        print(f"  Quality: {quality_rate:.1%}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        gc.collect()


def demo_export():
    """Demonstrate export functionality."""
    print("\n=== Export Functionality ===")
    
    indexer = safe_create_indexer()
    if not indexer:
        return
    
    try:
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)
        
        # Export as JSON
        json_path = output_dir / "exported_embeddings.json"
        if indexer.export_collection_data(str(json_path), "json"):
            print(f"✅ Exported to JSON: {json_path}")
        
        # Export as CSV
        csv_path = output_dir / "exported_embeddings.csv"
        if indexer.export_collection_data(str(csv_path), "csv"):
            print(f"✅ Exported to CSV: {csv_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        gc.collect()


def demo_indexing_pipeline(*args, **kwargs):
    """Complete indexing pipeline demonstration."""
    print("🚀 Complete Indexing Pipeline Demo")
    print("=" * 50)
    
    # Check dependencies
    try:
        import sentence_transformers
        import chromadb
        print("✅ Dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return
    
    if not EMBEDDING_MODULE_AVAILABLE:
        print("❌ Embedding module not available")
        return
    
    # Run pipeline steps with safety
    steps = [
        ("Data Validation", demo_validation),
        ("Embedding Model", demo_embedding_model),
        ("Indexing", demo_indexing),
        ("Semantic Search", demo_search),
        ("Metadata", demo_metadata),
        ("Batch Processing", demo_batch_processing),
        ("Topic Search", demo_topic_search),
        ("Export", demo_export)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        try:
            step_func()
            # Force garbage collection between steps
            gc.collect()
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("🏁 Pipeline completed!")


def main():
    """Run all demo functions."""
    print("🚀 Financial Complaint Embedding and Indexing Demo")
    print("=" * 50)
    
    # Check dependencies
    try:
        import sentence_transformers
        import chromadb
        print("✅ Dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return
    
    if not EMBEDDING_MODULE_AVAILABLE:
        print("❌ Embedding module not available")
        return
    
    # Run demos with safety
    demos = [
        ("Embedding Model", demo_embedding_model),
        ("Data Validation", demo_validation),
        ("Indexing", demo_indexing),
        ("Semantic Search", demo_search),
        ("Metadata", demo_metadata),
        ("Batch Processing", demo_batch_processing),
        ("Topic Search", demo_topic_search),
        ("Export", demo_export)
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            demo_func()
            # Force garbage collection between demos
            gc.collect()
        except Exception as e:
            print(f"❌ Error in {demo_name}: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("🏁 Demo completed!")


if __name__ == "__main__":
    main() 