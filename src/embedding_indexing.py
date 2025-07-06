"""
Embedding and Indexing Module for Financial Complaint Analysis

This module provides embedding and indexing capabilities for financial
complaint text chunks using sentence-transformers/all-MiniLM-L6-v2 and ChromaDB.
"""

import logging
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import os

# Embedding model imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# FAISS imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ComplaintEmbeddingIndexer:
    """
    Embedding and indexing class for financial complaint analysis using FAISS.

    Example:
        >>> indexer = ComplaintEmbeddingIndexer()
        >>> indexer.index_chunks(chunked_narratives)
        >>> results = indexer.search_similar_chunks('fraudulent charges', n_results=10)
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 faiss_index_path: 'str | None' = None):
        """
        Initialize the complaint embedding indexer with FAISS.

        Args:
            model_name (str): Name of the sentence-transformers model to use.
            faiss_index_path (str, optional): Path to a saved FAISS index.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required")
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required")
        
        self.model_name = model_name
        self.faiss_index_path = faiss_index_path
        self._initialize_embedding_model()
        self._initialize_vector_store()
        logger.info(f"Initialized ComplaintEmbeddingIndexer with model: {model_name}")
    
    def _initialize_embedding_model(self) -> None:
        """
        Initialize the sentence transformer model and set embedding dimension.
        """
        self.embedding_model = SentenceTransformer(self.model_name)
        test_embedding = self.embedding_model.encode("Test text")
        self.embedding_dimension = len(test_embedding)
        logger.info(f"Loaded model with {self.embedding_dimension} dimensions")
    
    def _initialize_vector_store(self) -> None:
        """
        Initialize the FAISS vector store and metadata storage.
        """
        # Use L2 index for now; can be changed to other types if needed
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.texts = []
        self.metadatas = []
        self.ids = []
    
    def generate_embeddings(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (list[str]): List of texts to embed.
            show_progress (bool): Whether to show a progress bar.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        if not texts:
            return []
        
        valid_texts = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
        if not valid_texts:
            return []
        
        try:
            # Process in smaller batches to avoid memory issues
            batch_size = 32  # Smaller batch size for stability
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                try:
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts, 
                        show_progress_bar=show_progress,
                        convert_to_numpy=True
                    )
                    all_embeddings.extend(batch_embeddings.tolist())
                except Exception as batch_error:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {batch_error}")
                    # Continue with next batch instead of failing completely
                    continue
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def index_chunks(self, chunked_narratives: list[dict]) -> bool:
        """
        Index chunked narratives with their embeddings and metadata using FAISS.

        Args:
            chunked_narratives (list[dict]): List of chunk dictionaries with 'chunk_text' and metadata.

        Returns:
            bool: True if indexing was successful, False otherwise.
        """
        if not chunked_narratives:
            return False
        
        texts, metadatas, ids = [], [], []
        
        for i, chunk in enumerate(chunked_narratives):
            if not isinstance(chunk, dict) or 'chunk_text' not in chunk:
                continue
            
            text = chunk.get('chunk_text', '').strip()
            if not text:
                continue
            
            texts.append(text)
            
            metadata = {
                'original_narrative_index': chunk.get('original_narrative_index', i),
                'chunk_index': chunk.get('chunk_index', 0),
                'chunk_length': chunk.get('chunk_length', len(text)),
                'chunk_word_count': chunk.get('chunk_word_count', len(text.split())),
                'chunking_method': chunk.get('chunking_method', 'unknown'),
                'indexed_at': datetime.now().isoformat()
            }
            
            # Add original metadata
            for key, value in chunk.items():
                if key not in ['chunk_text', 'original_narrative_index', 'chunk_index', 
                             'chunk_length', 'chunk_word_count', 'chunking_method']:
                    metadata[f'original_{key}'] = str(value) if value is not None else 'null'
            
            metadatas.append(metadata)
            ids.append(f"chunk_{metadata['original_narrative_index']}_{metadata['chunk_index']}")
        
        if not texts:
            return False
        
        embeddings = self.generate_embeddings(texts, show_progress=False)
        if not embeddings:
            return False
        
        try:
            emb_array = np.array(embeddings).astype('float32')
            self.index.add(emb_array)
            self.texts.extend(texts)
            self.metadatas.extend(metadatas)
            self.ids.extend(ids)
            logger.info(f"Indexed {len(texts)} chunks in FAISS")
            return True
        except Exception as e:
            logger.error(f"Error indexing chunks in FAISS: {e}")
            return False
    
    def search_similar_chunks(self, 
                              query: str, 
                              n_results: int = 10,
                              filter_metadata: 'dict | None' = None) -> list[dict]:
        """
        Search for similar chunks using FAISS.

        Args:
            query (str): Query string to search for similar chunks.
            n_results (int): Number of results to return.
            filter_metadata (dict, optional): Metadata to filter results.

        Returns:
            list[dict]: List of result dictionaries with id, text, metadata, and distance.
        """
        if not query.strip() or self.index.ntotal == 0:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query]).astype('float32')
            D, I = self.index.search(query_embedding, n_results)
            results = []
            for idx, dist in zip(I[0], D[0]):
                if idx < 0 or idx >= len(self.texts):
                    continue
                # Optionally filter by metadata
                if filter_metadata:
                    meta = self.metadatas[idx]
                    if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                        continue
                results.append({
                    'id': self.ids[idx],
                    'text': self.texts[idx],
                    'metadata': self.metadatas[idx],
                    'distance': float(dist)
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching chunks in FAISS: {e}")
            return []
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the indexed collection.

        Returns:
            dict: Dictionary with total_chunks, embedding_dimension, and model_name.
        """
        try:
            count = self.index.ntotal
            return {
                'total_chunks': count,
                'embedding_dimension': self.embedding_dimension,
                'model_name': self.model_name
            }
        except Exception as e:
            logger.error(f"Error getting FAISS stats: {e}")
            return {}
    
    def export_collection_data(self, output_path: str, format: str = "json") -> bool:
        """
        Export the collection data to a file.

        Args:
            output_path (str): Path to the output file.
            format (str): Format to export ('json' or 'csv').

        Returns:
            bool: True if export was successful, False otherwise.
        """
        try:
            if format.lower() == "json":
                export_data = {
                    'ids': self.ids,
                    'documents': self.texts,
                    'metadatas': self.metadatas,
                    'export_info': {
                        'exported_at': datetime.now().isoformat(),
                        'total_chunks': len(self.ids),
                        'model_name': self.model_name
                    }
                }
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            elif format.lower() == "csv":
                data = []
                for i in range(len(self.ids)):
                    row = {'id': self.ids[i], 'text': self.texts[i], **self.metadatas[i]}
                    data.append(row)
                pd.DataFrame(data).to_csv(output_path, index=False, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported format: {format}")
            logger.info(f"Exported data to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the FAISS index and metadata lists.

        Returns:
            bool: True if cleared successfully, False otherwise.
        """
        try:
            self.index.reset()
            self.texts = []
            self.metadatas = []
            self.ids = []
            logger.info("Cleared FAISS index and metadata lists")
            return True
        except Exception as e:
            logger.error(f"Error clearing FAISS index: {e}")
            return False

    def save_faiss_index(self, directory: str = "vector_store") -> bool:
        """
        Save the FAISS index and metadata to the specified directory.

        Args:
            directory (str): Directory to save the index and metadata.

        Returns:
            bool: True if saved successfully, False otherwise.
        """
        os.makedirs(directory, exist_ok=True)  # Ensure directory exists
        index_path = os.path.join(directory, "faiss_index.bin")
        meta_path = os.path.join(directory, "faiss_metadata.json")
        try:
            faiss.write_index(self.index, index_path)
            # Save metadata and ids
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "texts": self.texts,
                    "metadatas": self.metadatas,
                    "ids": self.ids,
                    "model_name": self.model_name,
                    "embedding_dimension": self.embedding_dimension
                }, f)
            logger.info(f"Saved FAISS index and metadata to {directory}")
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            return False

    def load_faiss_index(self, directory: str = "vector_store") -> bool:
        """
        Load the FAISS index and metadata from the specified directory.

        Args:
            directory (str): Directory to load the index and metadata from.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        index_path = os.path.join(directory, "faiss_index.bin")
        meta_path = os.path.join(directory, "faiss_metadata.json")
        try:
            self.index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.texts = meta.get("texts", [])
                self.metadatas = meta.get("metadatas", [])
                self.ids = meta.get("ids", [])
                self.model_name = meta.get("model_name", self.model_name)
                self.embedding_dimension = meta.get("embedding_dimension", self.embedding_dimension)
            logger.info(f"Loaded FAISS index and metadata from {directory}")
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False


def create_embedding_indexer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                           faiss_index_path: 'str | None' = None) -> ComplaintEmbeddingIndexer:
    """
    Factory function to create a ComplaintEmbeddingIndexer instance.

    Args:
        model_name (str): Name of the sentence-transformers model to use.
        faiss_index_path (str, optional): Path to a saved FAISS index.

    Returns:
        ComplaintEmbeddingIndexer: Instance of the embedding indexer.
    """
    return ComplaintEmbeddingIndexer(model_name, faiss_index_path)


def batch_index_from_csv(csv_path: str, 
                        indexer: ComplaintEmbeddingIndexer,
                        batch_size: int = 1000) -> bool:
    """
    Batch index chunks from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing chunked narratives.
        indexer (ComplaintEmbeddingIndexer): The embedding indexer instance.
        batch_size (int): Number of records to process per batch.

    Returns:
        bool: True if at least 80% of chunks were indexed successfully, False otherwise.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return False
    
    try:
        # Try different encodings
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None or df.empty or 'chunk_text' not in df.columns:
            logger.error("Invalid CSV file")
            return False
        
        chunks = df.to_dict('records')
        total_chunks = len(chunks)
        success_count = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            if indexer.index_chunks(batch):
                success_count += len(batch)
        
        success_rate = success_count / total_chunks if total_chunks > 0 else 0
        logger.info(f"Batch indexing completed: {success_count}/{total_chunks} chunks")
        return success_rate >= 0.8
        
    except Exception as e:
        logger.error(f"Error in batch indexing: {e}")
        return False


def search_complaints_by_topic(query: str,
                             indexer: ComplaintEmbeddingIndexer,
                             n_results: int = 10,
                             product_category: 'str | None' = None) -> list[dict]:
    """
    Search for complaints by topic with optional filtering.

    Args:
        query (str): Query string to search for.
        indexer (ComplaintEmbeddingIndexer): The embedding indexer instance.
        n_results (int): Number of results to return.
        product_category (str, optional): Product category to filter results.

    Returns:
        list[dict]: List of result dictionaries.
    """
    filter_metadata = {"original_Product": product_category} if product_category else None
    return indexer.search_similar_chunks(query, n_results, filter_metadata)


def validate_chunk_data(chunks: list[dict]) -> dict:
    """
    Validate chunk data structure and quality.

    Args:
        chunks (list[dict]): List of chunk dictionaries.

    Returns:
        dict: Validation results including counts and issues.
    """
    validation_results = {
        'total_chunks': len(chunks),
        'valid_chunks': 0,
        'invalid_chunks': 0,
        'missing_text': 0,
        'empty_text': 0,
        'issues': []
    }
    
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            validation_results['invalid_chunks'] += 1
            continue
        
        if 'chunk_text' not in chunk:
            validation_results['missing_text'] += 1
            continue
        
        text = chunk.get('chunk_text', '').strip()
        if not text:
            validation_results['empty_text'] += 1
            continue
        
        validation_results['valid_chunks'] += 1
    
    return validation_results 