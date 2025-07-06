"""
Chunked Data Loading Demo

This script demonstrates how to load large CSV files using chunked reading
to avoid memory issues. This is particularly useful for the CFPB complaints
dataset which contains ~9.6 million rows.

Usage:
    python chunked_data_loading_demo.py

Features:
- Memory-efficient chunked CSV reading
- Configurable chunk size based on available memory
- Progress tracking during loading
- Memory usage monitoring
"""

import os
import sys
import pandas as pd
import psutil
import time
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from eda_preprocessing import ComplaintAnalyzer
    EDA_MODULE_AVAILABLE = True
except ImportError:
    print("Warning: eda_preprocessing module not available")
    EDA_MODULE_AVAILABLE = False


def log_memory_usage(context=""):
    """Log current memory usage."""
    try:
        memory = psutil.virtual_memory()
        print(f"[MEMORY] {context} | Available: {memory.available / (1024**3):.2f} GB | Used: {memory.percent}%")
    except Exception:
        pass


def load_complaints_chunked(filepath, chunk_size=100000):
    """
    Load complaints data using chunked reading to manage memory usage.
    
    Args:
        filepath (str): Path to the complaints CSV file
        chunk_size (int): Number of rows to read at a time
        
    Returns:
        pd.DataFrame: Concatenated DataFrame with all data
    """
    print(f"Loading data from: {filepath}")
    print(f"Chunk size: {chunk_size:,} rows")
    
    log_memory_usage("Before loading")
    
    start_time = time.time()
    chunks = []
    chunk_count = 0
    
    try:
        # Read the CSV file in chunks
        for chunk in pd.read_csv(filepath, low_memory=False, chunksize=chunk_size):
            chunk_count += 1
            chunks.append(chunk)
            
            # Log progress every 10 chunks
            if chunk_count % 10 == 0:
                elapsed = time.time() - start_time
                rows_loaded = chunk_count * chunk_size
                print(f"Loaded {chunk_count} chunks ({rows_loaded:,} rows) in {elapsed:.1f}s")
                log_memory_usage(f"After chunk {chunk_count}")
        
        print(f"Concatenating {len(chunks)} chunks...")
        dataset = pd.concat(chunks, ignore_index=True)
        
        total_time = time.time() - start_time
        print(f"Total loading time: {total_time:.1f}s")
        print(f"Final dataset shape: {dataset.shape}")
        log_memory_usage("After concatenation")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def load_complaints_with_processing(filepath, chunk_size=100000):
    """
    Load complaints data with optional processing during chunk loading.
    
    Args:
        filepath (str): Path to the complaints CSV file
        chunk_size (int): Number of rows to read at a time
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    print(f"Loading and processing data from: {filepath}")
    print(f"Chunk size: {chunk_size:,} rows")
    
    log_memory_usage("Before loading")
    
    start_time = time.time()
    chunks = []
    chunk_count = 0
    
    # Define target products for filtering
    target_products = [
        'Credit card',
        'Personal loan', 
        'Buy Now, Pay Later (BNPL)',
        'Savings account',
        'Money transfers'
    ]
    
    try:
        # Read the CSV file in chunks and process each chunk
        for chunk in pd.read_csv(filepath, low_memory=False, chunksize=chunk_size):
            chunk_count += 1
            
            # Process each chunk (filter by products and remove empty narratives)
            processed_chunk = chunk[
                (chunk['Product'].isin(target_products)) &
                (chunk['Consumer complaint narrative'].notna()) &
                (chunk['Consumer complaint narrative'].str.strip() != '')
            ].copy()
            
            if not processed_chunk.empty:
                chunks.append(processed_chunk)
            
            # Log progress every 10 chunks
            if chunk_count % 10 == 0:
                elapsed = time.time() - start_time
                rows_loaded = chunk_count * chunk_size
                total_filtered = sum(len(c) for c in chunks)
                print(f"Loaded {chunk_count} chunks, filtered to {total_filtered:,} rows in {elapsed:.1f}s")
                log_memory_usage(f"After chunk {chunk_count}")
        
        if chunks:
            print(f"Concatenating {len(chunks)} processed chunks...")
            dataset = pd.concat(chunks, ignore_index=True)
        else:
            print("No data remained after filtering")
            dataset = pd.DataFrame()
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.1f}s")
        print(f"Final dataset shape: {dataset.shape}")
        log_memory_usage("After concatenation")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def demo_basic_loading():
    """Demonstrate basic chunked loading."""
    print("=== BASIC CHUNKED LOADING DEMO ===")
    
    # Find the data file
    data_path = Path(__file__).parent.parent / "data" / "complaints.csv"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return None
    
    # Load with chunked reading
    chunk_size = 100000  # Adjust based on available memory
    dataset = load_complaints_chunked(str(data_path), chunk_size)
    
    if dataset is not None:
        print(f"\nSuccessfully loaded {len(dataset):,} complaints")
        print(f"Columns: {list(dataset.columns)}")
        print(f"Sample data:")
        print(dataset.head())
    
    return dataset


def demo_processing_during_loading():
    """Demonstrate loading with processing during chunk reading."""
    print("\n=== PROCESSING DURING LOADING DEMO ===")
    
    # Find the data file
    data_path = Path(__file__).parent.parent / "data" / "complaints.csv"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return None
    
    # Load with processing during chunk reading
    chunk_size = 100000  # Adjust based on available memory
    dataset = load_complaints_with_processing(str(data_path), chunk_size)
    
    if dataset is not None and not dataset.empty:
        print(f"\nSuccessfully loaded and filtered {len(dataset):,} complaints")
        print(f"Product distribution:")
        print(dataset['Product'].value_counts())
        print(f"Sample data:")
        print(dataset.head())
    
    return dataset


def demo_with_analyzer():
    """Demonstrate using the ComplaintAnalyzer with chunked loading."""
    if not EDA_MODULE_AVAILABLE:
        print("Skipping analyzer demo - module not available")
        return
    
    print("\n=== COMPLAINT ANALYZER DEMO ===")
    
    # Find the data file
    data_path = Path(__file__).parent.parent / "data" / "complaints.csv"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    # Load data using the analyzer's chunked loading
    analyzer = ComplaintAnalyzer()
    dataset = analyzer.load_complaints_data(str(data_path), chunk_size=100000)
    
    if dataset is not None:
        print(f"\nLoaded {len(dataset):,} complaints using ComplaintAnalyzer")
        
        # Perform some basic analysis
        print("\n=== BASIC ANALYSIS ===")
        analyzer.initial_eda()
        
        # Show product distribution
        print("\n=== PRODUCT DISTRIBUTION ===")
        product_counts = dataset['Product'].value_counts()
        print(product_counts.head(10))


def main():
    """Main function to run all demos."""
    print("Chunked Data Loading Demo")
    print("=" * 50)
    
    # Check if data file exists
    data_path = Path(__file__).parent.parent / "data" / "complaints.csv"
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the complaints.csv file is in the data directory.")
        return
    
    # Run demos
    try:
        # Demo 1: Basic chunked loading
        dataset1 = demo_basic_loading()
        
        # Demo 2: Processing during loading
        dataset2 = demo_processing_during_loading()
        
        # Demo 3: Using ComplaintAnalyzer
        demo_with_analyzer()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        
        # Memory cleanup
        del dataset1, dataset2
        log_memory_usage("After cleanup")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 