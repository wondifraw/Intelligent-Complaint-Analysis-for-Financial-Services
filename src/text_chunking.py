"""
Text Chunking Module for Financial Complaint Analysis

This module provides comprehensive text chunking strategies for processing long
narrative texts in financial complaint datasets. It includes both LangChain-based
and custom chunking approaches with configurable parameters and robust error handling.

"""

import logging
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

# LangChain imports for text splitting
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.text_splitter import TokenTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Custom chunking methods will be used.")

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NarrativeChunkingStrategy:
    """
    A comprehensive text chunking strategy class for financial complaint narratives.
    
    This class provides multiple chunking approaches optimized for financial
    complaint text analysis, including both LangChain-based and custom methods.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 chunking_method: str = "recursive_character"):
        """
        Initialize the narrative chunking strategy.
        
        Args:
            chunk_size (int): Maximum number of characters per chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            chunking_method (str): Method to use for chunking ('recursive_character', 
                                 'character', 'token', 'custom_sentence', 'custom_paragraph')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = chunking_method
        
        self._validate_parameters()
        self._initialize_splitter()
    
    def _validate_parameters(self):
        """
        Validate the chunking parameters to ensure they are reasonable.
        
        Raises:
            ValueError: If parameters are invalid
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        if self.chunking_method not in [
            "recursive_character", "character", "token", 
            "custom_sentence", "custom_paragraph"
        ]:
            raise ValueError(f"Unsupported chunking method: {self.chunking_method}")
    
    def _initialize_splitter(self):
        """
        Initialize the appropriate chunking method based on the selected strategy.
        """
        try:
            if self.chunking_method == "recursive_character" and LANGCHAIN_AVAILABLE:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
                )
            elif self.chunking_method == "character" and LANGCHAIN_AVAILABLE:
                self.text_splitter = CharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            elif self.chunking_method == "token" and LANGCHAIN_AVAILABLE:
                self.text_splitter = TokenTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            else:
                # Use custom methods if LangChain is not available or custom method selected
                self.text_splitter = None
                
        except Exception as e:
            logger.error(f"Error initializing splitter: {e}")
            self.text_splitter = None
    
    def chunk_single_narrative(self, narrative_text: str) -> List[str]:
        """
        Chunk a single narrative text using the configured strategy.
        
        Args:
            narrative_text (str): The narrative text to chunk
            
        Returns:
            List[str]: List of text chunks
            
        Raises:
            ValueError: If narrative_text is None or empty
        """
        if not narrative_text or not isinstance(narrative_text, str):
            raise ValueError("narrative_text must be a non-empty string")
        
        try:
            # Clean the text first
            cleaned_text = self._preprocess_text(narrative_text)
            
            if not cleaned_text.strip():
                return []
            
            # Apply the appropriate chunking method
            if self.text_splitter is not None and LANGCHAIN_AVAILABLE:
                chunks = self._apply_langchain_chunking(cleaned_text)
            else:
                chunks = self._apply_custom_chunking(cleaned_text)
            
            # Post-process chunks
            processed_chunks = self._postprocess_chunks(chunks)
            
            logger.info(f"Successfully chunked narrative into {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error chunking single narrative: {e}")
            return []
    
    def chunk_narratives_batch(self, narratives: Union[List[str], pd.Series]) -> List[Dict]:
        """
        Chunk multiple narratives in batch with metadata tracking.
        
        Args:
            narratives (Union[List[str], pd.Series]): List or Series of narrative texts
            
        Returns:
            List[Dict]: List of dictionaries containing chunks with metadata
        """
        if not narratives:
            logger.warning("No narratives provided for batch chunking")
            return []
        
        try:
            chunked_narratives = []
            
            for idx, narrative in enumerate(narratives):
                try:
                    chunks = self.chunk_single_narrative(str(narrative))
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_metadata = {
                            "original_narrative_index": idx,
                            "chunk_index": chunk_idx,
                            "chunk_text": chunk,
                            "chunk_length": len(chunk),
                            "chunk_word_count": len(chunk.split()),
                            "chunking_method": self.chunking_method
                        }
                        chunked_narratives.append(chunk_metadata)
                        
                except Exception as e:
                    logger.error(f"Error processing narrative at index {idx}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(chunked_narratives)} chunks from {len(narratives)} narratives")
            return chunked_narratives
            
        except Exception as e:
            logger.error(f"Error in batch chunking: {e}")
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess narrative text before chunking.
        
        Args:
            text (str): Raw narrative text
            
        Returns:
            str: Preprocessed text
        """
        try:
            # Remove extra whitespace
            text = " ".join(text.split())
            
            # Normalize line breaks
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # Remove null characters
            text = text.replace('\x00', '')
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    def _apply_langchain_chunking(self, text: str) -> List[str]:
        """
        Apply LangChain-based chunking methods.
        
        Args:
            text (str): Preprocessed text
            
        Returns:
            List[str]: List of text chunks
        """
        try:
            if self.text_splitter is None:
                raise ValueError("LangChain text splitter not initialized")
            
            chunks = self.text_splitter.split_text(text)
            return [chunk.strip() for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            logger.error(f"Error applying LangChain chunking: {e}")
            return [text]  # Return original text as single chunk
    
    def _apply_custom_chunking(self, text: str) -> List[str]:
        """
        Apply custom chunking methods when LangChain is not available.
        
        Args:
            text (str): Preprocessed text
            
        Returns:
            List[str]: List of text chunks
        """
        try:
            if self.chunking_method == "custom_sentence":
                return self._sentence_chunking(text)
            elif self.chunking_method == "custom_paragraph":
                return self._paragraph_chunking(text)
            else:
                return self._character_chunking(text)
                
        except Exception as e:
            logger.error(f"Error applying custom chunking: {e}")
            return [text]
    
    def _sentence_chunking(self, text: str) -> List[str]:
        """
        Custom sentence-based chunking that preserves sentence boundaries.
        
        Args:
            text (str): Preprocessed text
            
        Returns:
            List[str]: List of sentence-based chunks
        """
        try:
            # Split by sentence endings
            sentences = []
            current_sentence = ""
            
            for char in text:
                current_sentence += char
                if char in '.!?' and len(current_sentence.strip()) > 0:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
            
            # Add any remaining text
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            
            # Group sentences into chunks
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in custom sentence chunking: {e}")
            return [text]
    
    def _paragraph_chunking(self, text: str) -> List[str]:
        """
        Custom paragraph-based chunking that preserves paragraph boundaries.
        
        Args:
            text (str): Preprocessed text
            
        Returns:
            List[str]: List of paragraph-based chunks
        """
        try:
            # Split by paragraph breaks
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) <= self.chunk_size:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = paragraph
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in custom paragraph chunking: {e}")
            return [text]
    
    def _character_chunking(self, text: str) -> List[str]:
        """
        Custom character-based chunking with overlap.
        
        Args:
            text (str): Preprocessed text
            
        Returns:
            List[str]: List of character-based chunks
        """
        try:
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + self.chunk_size
                
                # Try to break at word boundary
                if end < len(text):
                    # Look for the last space before the end
                    last_space = text.rfind(' ', start, end)
                    if last_space > start:
                        end = last_space
                
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                
                # Move start position with overlap
                start = end - self.chunk_overlap
                if start >= len(text):
                    break
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in custom character chunking: {e}")
            return [text]
    
    def _postprocess_chunks(self, chunks: List[str]) -> List[str]:
        """
        Post-process chunks to ensure quality and consistency.
        
        Args:
            chunks (List[str]): Raw chunks
            
        Returns:
            List[str]: Processed chunks
        """
        try:
            processed_chunks = []
            
            for chunk in chunks:
                # Remove chunks that are too short (likely incomplete)
                if len(chunk.strip()) < 50:  # Minimum meaningful chunk size
                    continue
                
                # Clean up the chunk
                cleaned_chunk = chunk.strip()
                if cleaned_chunk:
                    processed_chunks.append(cleaned_chunk)
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error in postprocessing chunks: {e}")
            return chunks
    
    def analyze_performance(self, chunked_narratives: List[Dict]) -> Dict:
        """
        Analyze the performance and characteristics of the chunking process.
        
        Args:
            chunked_narratives (List[Dict]): List of chunked narratives with metadata
            
        Returns:
            Dict: Performance metrics
        """
        try:
            if not chunked_narratives:
                return {}
            
            chunk_lengths = [chunk["chunk_length"] for chunk in chunked_narratives]
            word_counts = [chunk["chunk_word_count"] for chunk in chunked_narratives]
            
            return {
                "total_chunks": len(chunked_narratives),
                "avg_chunk_length": np.mean(chunk_lengths),
                "std_chunk_length": np.std(chunk_lengths),
                "min_chunk_length": np.min(chunk_lengths),
                "max_chunk_length": np.max(chunk_lengths),
                "avg_word_count": np.mean(word_counts),
                "std_word_count": np.std(word_counts),
                "chunking_method": self.chunking_method,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {}
    
    def save_results(self, chunked_narratives: List[Dict], output_path: str, format: str = "json") -> bool:
        """
        Save chunked narratives to file.
        
        Args:
            chunked_narratives (List[Dict]): List of chunked narratives
            output_path (str): Path to save the output file
            format (str): Output format ('json', 'csv', 'pickle')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(chunked_narratives, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == "csv":
                df = pd.DataFrame(chunked_narratives)
                df.to_csv(output_path, index=False, encoding='utf-8')
            
            elif format.lower() == "pickle":
                import pickle
                with open(output_path, 'wb') as f:
                    pickle.dump(chunked_narratives, f)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Successfully saved chunked narratives to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False


def create_optimal_strategy(narrative_lengths: List[int], target_size: int = 1000, overlap_ratio: float = 0.2) -> NarrativeChunkingStrategy:
    """
    Create an optimal chunking strategy based on narrative length analysis.
    
    Args:
        narrative_lengths (List[int]): List of narrative lengths in characters
        target_size (int): Target chunk size in characters
        overlap_ratio (float): Ratio of overlap (0.0 to 1.0)
        
    Returns:
        NarrativeChunkingStrategy: Optimized chunking strategy
    """
    try:
        if not narrative_lengths:
            raise ValueError("narrative_lengths cannot be empty")
        
        # Analyze narrative length distribution
        avg_length = np.mean(narrative_lengths)
        median_length = np.median(narrative_lengths)
        std_length = np.std(narrative_lengths)
        
        # Adjust chunk size based on narrative characteristics
        if avg_length < target_size * 0.5:
            # Most narratives are short, use smaller chunks
            optimal_size = max(500, int(avg_length * 1.5))
        elif avg_length > target_size * 2:
            # Most narratives are very long, use larger chunks
            optimal_size = min(2000, int(avg_length * 0.8))
        else:
            # Narratives are moderate length, use target size
            optimal_size = target_size
        
        # Calculate overlap
        optimal_overlap = int(optimal_size * overlap_ratio)
        
        # Choose chunking method based on length variability
        if std_length / avg_length > 0.5:
            # High variability, use recursive character splitting
            method = "recursive_character"
        else:
            # Low variability, use sentence-based splitting
            method = "custom_sentence"
        
        logger.info(f"Created optimal chunking strategy: size={optimal_size}, "
                   f"overlap={optimal_overlap}, method={method}")
        
        return NarrativeChunkingStrategy(
            chunk_size=optimal_size,
            chunk_overlap=optimal_overlap,
            chunking_method=method
        )
        
    except Exception as e:
        logger.error(f"Error creating optimal chunking strategy: {e}")
        # Return default strategy
        return NarrativeChunkingStrategy()


def experiment_parameters(narratives: List[str], 
                        chunk_sizes: List[int] = [500, 1000, 1500],
                        overlap_ratios: List[float] = [0.1, 0.2, 0.3],
                        methods: List[str] = ["recursive_character", "custom_sentence"]) -> Dict:
    """
    Experiment with different chunking parameters to find optimal settings.
    
    Args:
        narratives (List[str]): Sample narratives for experimentation
        chunk_sizes (List[int]): List of chunk sizes to test
        overlap_ratios (List[float]): List of overlap ratios to test
        methods (List[str]): List of chunking methods to test
        
    Returns:
        Dict: Results of the experimentation with performance metrics
    """
    try:
        results = []
        
        for method in methods:
            for chunk_size in chunk_sizes:
                for overlap_ratio in overlap_ratios:
                    try:
                        # Create strategy with current parameters
                        strategy = NarrativeChunkingStrategy(
                            chunk_size=chunk_size,
                            chunk_overlap=int(chunk_size * overlap_ratio),
                            chunking_method=method
                        )
                        
                        # Test on sample narratives
                        chunked_results = strategy.chunk_narratives_batch(narratives[:10])  # Use first 10 for testing
                        performance = strategy.analyze_performance(chunked_results)
                        
                        # Add parameter information
                        performance.update({
                            "method": method,
                            "chunk_size": chunk_size,
                            "overlap_ratio": overlap_ratio,
                            "overlap_size": int(chunk_size * overlap_ratio)
                        })
                        
                        results.append(performance)
                        
                    except Exception as e:
                        logger.error(f"Error testing parameters {method}, {chunk_size}, {overlap_ratio}: {e}")
                        continue
        
        # Find best performing configuration
        if results:
            best_result = max(results, key=lambda x: x.get("total_chunks", 0))
            logger.info(f"Best configuration: {best_result}")
        
        return {
            "all_results": results,
            "best_configuration": best_result if results else None
        }
        
    except Exception as e:
        logger.error(f"Error in chunking parameter experimentation: {e}")
        return {"all_results": [], "best_configuration": None} 