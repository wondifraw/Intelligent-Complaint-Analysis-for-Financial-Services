"""Demo script for text chunking module."""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))

from text_chunking import (
    NarrativeChunkingStrategy,
    create_optimal_strategy,
    experiment_parameters
)


def load_complaints_data():
    """Load financial complaint data from CSV file using chunked reading."""
    data_path = Path(__file__).parent.parent / "data" / "complaints_processed.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        # Load the CSV file using chunked reading to handle memory issues
        chunk_size = 100000  # Adjust this number based on your available memory
        print(f"Loading data in chunks of {chunk_size} rows...")
        
        chunks = []
        for chunk in pd.read_csv(data_path, low_memory=False, chunksize=chunk_size):
            # Optionally process each chunk here (e.g., filter, clean, etc.)
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(df)} complaints from {data_path}")
        
        # Check if the narrative column exists
        narrative_column = 'Consumer complaint narrative'
        if narrative_column not in df.columns:
            # Try alternative column names
            alternative_columns = ['Consumer complaint narrative_cleaned', 'narrative', 'text']
            for col in alternative_columns:
                if col in df.columns:
                    narrative_column = col
                    break
            else:
                raise ValueError(f"Could not find narrative column. Available columns: {list(df.columns)}")
        
        print(f"Using column: {narrative_column}")
        
        # Remove rows with empty narratives
        df = df.dropna(subset=[narrative_column])
        df = df[df[narrative_column].str.strip() != '']
        
        print(f"After filtering empty narratives: {len(df)} complaints")
        
        return df, narrative_column
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None





def demo_basic_chunking():
    """Demonstrate basic text chunking."""
    print("=== Basic Text Chunking ===")
    
    strategy = NarrativeChunkingStrategy(chunk_size=300, chunk_overlap=50)
    narratives = [
        "This is a long narrative that needs to be chunked into smaller pieces for better analysis and processing. It contains multiple sentences and should be split appropriately."
    ]
    
    results = strategy.chunk_narratives_batch(narratives)
    print(f"Created {len(results)} chunks")
    
    for i, chunk in enumerate(results[:2]):
        print(f"Chunk {i+1}: {chunk['chunk_text'][:80]}...")


def demo_chunking_methods():
    """Demonstrate different chunking methods."""
    print("\n=== Different Chunking Methods ===")
    
    text = "This is a test narrative. It has multiple sentences. Each sentence should be preserved. This helps maintain context."
    
    methods = ["recursive_character", "custom_sentence", "custom_paragraph"]
    
    for method in methods:
        strategy = NarrativeChunkingStrategy(chunk_size=100, chunk_overlap=20, chunking_method=method)
        chunks = strategy.chunk_single_narrative(text)
        print(f"{method}: {len(chunks)} chunks")


def demo_parameter_experimentation():
    """Demonstrate parameter experimentation."""
    print("\n=== Parameter Experimentation ===")
    
    narratives = [
        "Short narrative for testing.",
        "This is a longer narrative that contains more text and should be chunked into multiple pieces."
    ]
    
    results = experiment_parameters(
        narratives,
        chunk_sizes=[50, 100],
        overlap_ratios=[0.1, 0.2],
        methods=["custom_sentence"]
    )
    
    if results['best_configuration']:
        best = results['best_configuration']
        print(f"Best: {best['method']}, size={best['chunk_size']}, overlap={best['overlap_ratio']}")


def demo_optimal_strategy():
    """Demonstrate optimal strategy creation."""
    print("\n=== Optimal Strategy Creation ===")
    
    narrative_lengths = [500, 1000, 1500, 2000]
    strategy = create_optimal_strategy(narrative_lengths, target_size=1000, overlap_ratio=0.2)
    
    print(f"Optimal: {strategy.chunking_method}, size={strategy.chunk_size}, overlap={strategy.chunk_overlap}")


def demo_integration():
    """Demonstrate integration with actual complaint data."""
    print("\n=== Integration with Actual Complaint Data ===")
    
    # Load actual data
    df, narrative_column = load_complaints_data()
    
    if df is None:
        print("Error: Could not load complaint data. Please ensure the CSV file exists.")
        return
    
    # Use a subset for demo to avoid memory issues
    sample_size = min(100, len(df))
    df_sample = df.head(sample_size)
    narratives = df_sample[narrative_column].tolist()
    
    print(f"Processing {len(narratives)} narratives...")
    
    # Create chunking strategy
    strategy = NarrativeChunkingStrategy(chunk_size=400, chunk_overlap=100)
    results = strategy.chunk_narratives_batch(narratives)
    
    # Analyze performance
    performance = strategy.analyze_performance(results)
    print(f"Processed {len(narratives)} narratives into {len(results)} chunks")
    print(f"Average chunk length: {performance['avg_chunk_length']:.1f} characters")
    print(f"Average word count: {performance['avg_word_count']:.1f} words")
    
    # Save results as CSV
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    csv_output_path = output_dir / "chunked_narratives.csv"
    success = strategy.save_results(results, csv_output_path, "csv")
    print(f"Saved chunked results to CSV: {success}")
    print(f"Output file: {csv_output_path}")
    
    # Also save a summary of the chunking process
    summary_data = {
        'metric': list(performance.keys()),
        'value': list(performance.values())
    }
    summary_df = pd.DataFrame(summary_data)
    return summary_df

