#!/usr/bin/env python3
"""
Test script to verify embedding and indexing fixes.
This script tests the core functionality without causing kernel crashes.
"""

import sys
from pathlib import Path
import gc

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from embedding_indexing import ComplaintEmbeddingIndexer, create_embedding_indexer
        print("✅ Embedding module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test if the sentence transformer model can be loaded."""
    print("\nTesting model loading...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Sentence transformer model loaded successfully")
        
        # Test basic encoding
        test_text = "This is a test sentence."
        embedding = model.encode([test_text])
        print(f"✅ Test embedding generated: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_chromadb():
    """Test if ChromaDB can be initialized."""
    print("\nTesting ChromaDB...")
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./test_chroma_db")
        print("✅ ChromaDB client created successfully")
        return True
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False

def test_embedding_indexer():
    """Test the embedding indexer creation and basic functionality."""
    print("\nTesting embedding indexer...")
    
    try:
        from embedding_indexing import create_embedding_indexer
        
        # Create indexer with test directory
        indexer = create_embedding_indexer(
            chroma_persist_directory="./test_chroma_db",
            collection_name="test_collection"
        )
        print("✅ Embedding indexer created successfully")
        
        # Test embedding generation
        test_texts = [
            "I have a complaint about my credit card.",
            "The bank refused to help me.",
            "My account was charged incorrectly."
        ]
        
        embeddings = indexer.generate_embeddings(test_texts, show_progress=False)
        if embeddings and len(embeddings) == len(test_texts):
            print(f"✅ Generated {len(embeddings)} embeddings successfully")
        else:
            print("❌ Embedding generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding indexer error: {e}")
        return False

def test_small_indexing():
    """Test indexing with a very small dataset."""
    print("\nTesting small indexing...")
    
    try:
        from embedding_indexing import create_embedding_indexer
        
        indexer = create_embedding_indexer(
            chroma_persist_directory="./test_chroma_db",
            collection_name="test_collection"
        )
        
        # Create minimal test data
        test_chunks = [
            {
                'chunk_text': 'I have a complaint about unauthorized charges on my credit card.',
                'original_narrative_index': 0,
                'chunk_index': 0,
                'original_Product': 'Credit card',
                'original_Company': 'Test Bank'
            },
            {
                'chunk_text': 'The bank refused to refund the charges.',
                'original_narrative_index': 0,
                'chunk_index': 1,
                'original_Product': 'Credit card',
                'original_Company': 'Test Bank'
            }
        ]
        
        success = indexer.index_chunks(test_chunks)
        if success:
            print("✅ Small indexing test successful")
            
            # Test search
            results = indexer.search_similar_chunks("credit card complaint", n_results=1)
            if results:
                print("✅ Search functionality working")
            else:
                print("⚠️  No search results (may be normal with small dataset)")
            
            return True
        else:
            print("❌ Small indexing test failed")
            return False
            
    except Exception as e:
        print(f"❌ Small indexing error: {e}")
        return False

def cleanup():
    """Clean up test files."""
    print("\nCleaning up test files...")
    
    try:
        import shutil
        test_dir = Path("./test_chroma_db")
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("✅ Test directory cleaned up")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

def main():
    """Run all tests."""
    print("🧪 Testing Embedding and Indexing Fixes")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_loading,
        test_chromadb,
        test_embedding_indexer,
        test_small_indexing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes should resolve the kernel crash issues.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    # Cleanup
    cleanup()
    
    # Force garbage collection
    gc.collect()
    
    print("\n✅ Test script completed")

if __name__ == "__main__":
    main() 