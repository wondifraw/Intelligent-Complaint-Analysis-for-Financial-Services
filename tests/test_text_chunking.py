"""Test suite for the text chunking module."""

import unittest
import tempfile
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from text_chunking import (
    NarrativeChunkingStrategy,
    create_optimal_strategy,
    experiment_parameters
)


class TestNarrativeChunkingStrategy(unittest.TestCase):
    """Test cases for NarrativeChunkingStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_narrative = """
        I have been experiencing issues with my credit card account for the past three months. 
        The bank has been charging me unauthorized fees and has not responded to my multiple 
        complaints. I have called customer service over 10 times and each time I am told 
        that the issue will be resolved, but nothing happens.
        """
    
    def test_initialization(self):
        """Test initialization with valid parameters."""
        strategy = NarrativeChunkingStrategy(
            chunk_size=1000,
            chunk_overlap=200,
            chunking_method="recursive_character"
        )
        
        self.assertEqual(strategy.chunk_size, 1000)
        self.assertEqual(strategy.chunk_overlap, 200)
        self.assertEqual(strategy.chunking_method, "recursive_character")
    
    def test_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        with self.assertRaises(ValueError):
            NarrativeChunkingStrategy(chunk_size=0)
        
        with self.assertRaises(ValueError):
            NarrativeChunkingStrategy(chunk_overlap=1000, chunk_size=500)
        
        with self.assertRaises(ValueError):
            NarrativeChunkingStrategy(chunking_method="invalid_method")
    
    def test_chunk_single_narrative(self):
        """Test chunking a single narrative."""
        strategy = NarrativeChunkingStrategy(chunk_size=200, chunk_overlap=50)
        chunks = strategy.chunk_single_narrative(self.sample_narrative)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk.strip()), 0)
    
    def test_chunk_narratives_batch(self):
        """Test batch chunking of multiple narratives."""
        strategy = NarrativeChunkingStrategy(chunk_size=200, chunk_overlap=50)
        narratives = [self.sample_narrative, "Short complaint."]
        
        results = strategy.chunk_narratives_batch(narratives)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for result in results:
            self.assertIn("original_narrative_index", result)
            self.assertIn("chunk_text", result)
            self.assertIn("chunk_length", result)
    
    def test_custom_sentence_chunking(self):
        """Test custom sentence-based chunking."""
        strategy = NarrativeChunkingStrategy(
            chunk_size=200,
            chunk_overlap=50,
            chunking_method="custom_sentence"
        )
        
        chunks = strategy.chunk_single_narrative(self.sample_narrative)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    def test_analyze_performance(self):
        """Test performance analysis."""
        strategy = NarrativeChunkingStrategy(chunk_size=200, chunk_overlap=50)
        narratives = [self.sample_narrative]
        
        chunked_results = strategy.chunk_narratives_batch(narratives)
        performance = strategy.analyze_performance(chunked_results)
        
        self.assertIsInstance(performance, dict)
        self.assertIn("total_chunks", performance)
        self.assertIn("avg_chunk_length", performance)
    
    def test_save_results(self):
        """Test saving results to file."""
        strategy = NarrativeChunkingStrategy(chunk_size=200, chunk_overlap=50)
        narratives = [self.sample_narrative]
        chunked_results = strategy.chunk_narratives_batch(narratives)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            success = strategy.save_results(chunked_results, temp_path, "json")
            self.assertTrue(success)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            self.assertIsInstance(saved_data, list)
            self.assertEqual(len(saved_data), len(chunked_results))
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_create_optimal_strategy(self):
        """Test creating optimal chunking strategy."""
        narrative_lengths = [500, 1000, 1500, 2000]
        
        strategy = create_optimal_strategy(
            narrative_lengths,
            target_size=1000,
            overlap_ratio=0.2
        )
        
        self.assertIsInstance(strategy, NarrativeChunkingStrategy)
        self.assertGreater(strategy.chunk_size, 0)
        self.assertGreaterEqual(strategy.chunk_overlap, 0)
    
    def test_experiment_parameters(self):
        """Test parameter experimentation."""
        sample_narratives = [
            "This is a short narrative for testing.",
            "This is a longer narrative that contains more text and should be chunked."
        ]
        
        results = experiment_parameters(
            sample_narratives,
            chunk_sizes=[100, 200],
            overlap_ratios=[0.1, 0.2],
            methods=["custom_sentence"]
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn("all_results", results)
        self.assertIn("best_configuration", results)


if __name__ == "__main__":
    unittest.main(verbosity=2) 