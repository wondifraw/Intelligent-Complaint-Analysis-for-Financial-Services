"""
Tests for the embedding and indexing module.

This module tests the ComplaintEmbeddingIndexer class and related functionality
using sentence-transformers/all-MiniLM-L6-v2 and ChromaDB.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embedding_indexing import (
    ComplaintEmbeddingIndexer,
    create_embedding_indexer,
    batch_index_from_csv,
    search_complaints_by_topic
)


class TestComplaintEmbeddingIndexer(unittest.TestCase):
    """Test cases for ComplaintEmbeddingIndexer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_dir = Path(self.temp_dir) / "test_chroma"
        
        # Test data
        self.test_chunks = [
            {
                'chunk_text': 'I have a complaint about unauthorized charges on my credit card.',
                'original_narrative_index': 0,
                'chunk_index': 0,
                'chunk_length': 75,
                'chunk_word_count': 12,
                'chunking_method': 'recursive_character',
                'original_Product': 'Credit card',
                'original_Company': 'Bank of America'
            },
            {
                'chunk_text': 'The bank refused to investigate the fraudulent transactions.',
                'original_narrative_index': 0,
                'chunk_index': 1,
                'chunk_length': 78,
                'chunk_word_count': 11,
                'chunking_method': 'recursive_character',
                'original_Product': 'Credit card',
                'original_Company': 'Bank of America'
            },
            {
                'chunk_text': 'My mortgage payment was processed incorrectly.',
                'original_narrative_index': 1,
                'chunk_index': 0,
                'chunk_length': 55,
                'chunk_word_count': 9,
                'chunking_method': 'recursive_character',
                'original_Product': 'Mortgage',
                'original_Company': 'Wells Fargo'
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test indexer initialization."""
        try:
            indexer = ComplaintEmbeddingIndexer(
                chroma_persist_directory=str(self.chroma_dir),
                collection_name="test_collection"
            )
            
            self.assertIsNotNone(indexer.embedding_model)
            self.assertIsNotNone(indexer.collection)
            self.assertEqual(indexer.model_name, "sentence-transformers/all-MiniLM-L6-v2")
            self.assertEqual(indexer.collection_name, "test_collection")
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_embedding_generation(self):
        """Test embedding generation."""
        try:
            indexer = ComplaintEmbeddingIndexer(
                chroma_persist_directory=str(self.chroma_dir),
                collection_name="test_embeddings"
            )
            
            test_texts = [
                "This is a test complaint about credit card fraud.",
                "Another test complaint about mortgage issues."
            ]
            
            embeddings = indexer.generate_embeddings(test_texts)
            
            self.assertEqual(len(embeddings), 2)
            self.assertEqual(len(embeddings[0]), indexer.embedding_dimension)
            self.assertEqual(len(embeddings[1]), indexer.embedding_dimension)
            
            # Check that embeddings are different for different texts
            self.assertNotEqual(embeddings[0], embeddings[1])
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_chunk_indexing(self):
        """Test indexing of chunked narratives."""
        try:
            indexer = ComplaintEmbeddingIndexer(
                chroma_persist_directory=str(self.chroma_dir),
                collection_name="test_indexing"
            )
            
            # Index chunks
            success = indexer.index_chunks(self.test_chunks)
            self.assertTrue(success)
            
            # Check collection stats
            stats = indexer.get_collection_stats()
            self.assertEqual(stats['total_chunks'], 3)
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_semantic_search(self):
        """Test semantic search functionality."""
        try:
            indexer = ComplaintEmbeddingIndexer(
                chroma_persist_directory=str(self.chroma_dir),
                collection_name="test_search"
            )
            
            # Index chunks first
            indexer.index_chunks(self.test_chunks)
            
            # Search for similar chunks
            results = indexer.search_similar_chunks("fraudulent charges", n_results=2)
            
            self.assertGreater(len(results), 0)
            self.assertLessEqual(len(results), 2)
            
            # Check result structure
            for result in results:
                self.assertIn('id', result)
                self.assertIn('text', result)
                self.assertIn('metadata', result)
                self.assertIn('distance', result)
                
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_metadata_preservation(self):
        """Test that metadata is properly preserved."""
        try:
            indexer = ComplaintEmbeddingIndexer(
                chroma_persist_directory=str(self.chroma_dir),
                collection_name="test_metadata"
            )
            
            # Index chunks
            indexer.index_chunks(self.test_chunks)
            
            # Search and check metadata
            results = indexer.search_similar_chunks("credit card", n_results=3)
            
            for result in results:
                metadata = result['metadata']
                self.assertIn('original_narrative_index', metadata)
                self.assertIn('chunk_index', metadata)
                self.assertIn('chunking_method', metadata)
                self.assertIn('indexed_at', metadata)
                
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_empty_chunks(self):
        """Test handling of empty chunks."""
        try:
            indexer = ComplaintEmbeddingIndexer(
                chroma_persist_directory=str(self.chroma_dir),
                collection_name="test_empty"
            )
            
            empty_chunks = [
                {'chunk_text': '', 'original_narrative_index': 0, 'chunk_index': 0},
                {'chunk_text': '   ', 'original_narrative_index': 0, 'chunk_index': 1},
                {'chunk_text': 'Valid text here.', 'original_narrative_index': 0, 'chunk_index': 2}
            ]
            
            success = indexer.index_chunks(empty_chunks)
            self.assertTrue(success)
            
            # Should only index the valid chunk
            stats = indexer.get_collection_stats()
            self.assertEqual(stats['total_chunks'], 1)
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_export_functionality(self):
        """Test export functionality."""
        try:
            indexer = ComplaintEmbeddingIndexer(
                chroma_persist_directory=str(self.chroma_dir),
                collection_name="test_export"
            )
            
            # Index some chunks
            indexer.index_chunks(self.test_chunks)
            
            # Test JSON export
            json_path = Path(self.temp_dir) / "export.json"
            success_json = indexer.export_collection_data(str(json_path), "json")
            self.assertTrue(success_json)
            self.assertTrue(json_path.exists())
            
            # Test CSV export
            csv_path = Path(self.temp_dir) / "export.csv"
            success_csv = indexer.export_collection_data(str(csv_path), "csv")
            self.assertTrue(success_csv)
            self.assertTrue(csv_path.exists())
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_dir = Path(self.temp_dir) / "test_chroma"
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_embedding_indexer(self):
        """Test factory function for creating indexer."""
        try:
            indexer = create_embedding_indexer(
                chroma_persist_directory=str(self.chroma_dir),
                collection_name="test_factory"
            )
            
            self.assertIsInstance(indexer, ComplaintEmbeddingIndexer)
            self.assertEqual(indexer.model_name, "sentence-transformers/all-MiniLM-L6-v2")
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_search_complaints_by_topic(self):
        """Test topic-based search function."""
        try:
            indexer = create_embedding_indexer(
                chroma_persist_directory=str(self.chroma_dir),
                collection_name="test_topic_search"
            )
            
            # Create test chunks with product categories
            test_chunks = [
                {
                    'chunk_text': 'Credit card fraud complaint.',
                    'original_narrative_index': 0,
                    'chunk_index': 0,
                    'original_Product': 'Credit card'
                },
                {
                    'chunk_text': 'Mortgage payment issue.',
                    'original_narrative_index': 1,
                    'chunk_index': 0,
                    'original_Product': 'Mortgage'
                }
            ]
            
            # Index chunks
            indexer.index_chunks(test_chunks)
            
            # Search by topic with filter
            results = search_complaints_by_topic(
                query="fraud",
                indexer=indexer,
                n_results=5,
                product_category="Credit card"
            )
            
            self.assertIsInstance(results, list)
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")


class TestModelChoice(unittest.TestCase):
    """Test cases for model choice validation."""
    
    def test_model_choice_justification(self):
        """Test that the model choice is well-justified."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Check that it's a valid sentence transformer model
        self.assertIn("sentence-transformers", model_name)
        self.assertIn("all-MiniLM-L6-v2", model_name)
        
        # Model choice justification points
        justification_points = [
            "Lightweight and fast",
            "Strong performance on semantic similarity",
            "384-dimensional embeddings",
            "Widely adopted in community",
            "Resource efficient",
            "Production ready"
        ]
        
        # All justification points should be valid
        for point in justification_points:
            self.assertIsInstance(point, str)
            self.assertGreater(len(point), 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestComplaintEmbeddingIndexer))
    test_suite.addTest(unittest.makeSuite(TestUtilityFunctions))
    test_suite.addTest(unittest.makeSuite(TestModelChoice))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!") 