
# Intelligent Complaint Analysis for Financial Services

## Table of Contents
- [Summary](#summary)
- [Quickstart](#quickstart)
- [Project Structure](#project-structure)
- [Dependencies & Installation](#dependencies--installation)
- [Features](#features)
- [Workflow](#workflow)
- [Usage](#usage)
- [Data Processing Steps](#data-processing-steps)
- [Links & Resources](#links--resources)
- [Contributing](#contributing)
- [Contact](#contact)

---
## Summary
This project provides an end-to-end, modular pipeline for the intelligent analysis of consumer complaints in the financial services sector, using data from the Consumer Financial Protection Bureau (CFPB). The motivation is to help financial institutions, regulators, and researchers extract actionable insights from large volumes of unstructured complaint narratives, enabling better customer service, regulatory compliance, and risk management.

**Key Features and Workflow:**
- **Data Ingestion & EDA:** Robust loading and exploratory analysis of raw CFPB complaint data, including product distribution, narrative presence, and text length statistics.
- **Data Cleaning & Preprocessing:** Automated filtering for relevant products, removal of empty or low-quality narratives, and advanced text cleaning (boilerplate removal, normalization, etc.).
- **Text Chunking:** Flexible chunking strategies (recursive character, sentence, paragraph, and token-based) to break long complaint narratives into manageable, semantically meaningful pieces, preserving metadata for traceability.
- **Embedding Generation:** State-of-the-art sentence embeddings using transformer models (e.g., all-MiniLM-L6-v2) to convert text chunks into high-dimensional vectors suitable for semantic analysis.
- **Vector Indexing & Semantic Search:** Efficient similarity search and topic-based retrieval using FAISS and ChromaDB, supporting fast, scalable exploration of complaint themes and trends.
- **Batch Processing & Performance Analysis:** Tools for large-scale processing, chunking performance evaluation, and export of results for downstream analytics.
- **Jupyter Notebooks & Scripts:** Ready-to-use notebooks and demo scripts for each stage of the workflow, supporting both interactive exploration and automated pipelines.

**Practical Applications:**
- Rapidly identify emerging complaint topics, fraud patterns, or compliance risks.
- Enable semantic search for customer support teams to find similar cases and improve response quality.
- Support research into consumer protection, financial product issues, and regulatory trends.
- Provide a foundation for advanced analytics such as topic modeling, sentiment analysis, and automated complaint classification.

The codebase is designed for extensibility, reproducibility, and ease of use, with clear modular separation between EDA, preprocessing, chunking, embedding, and indexing components. Extensive documentation, tests, and example workflows are provided to accelerate adoption and adaptation for new datasets or use cases.

---

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Intelligent-Complaint-Analysis-for-Financial-Services.git
   cd Intelligent-Complaint-Analysis-for-Financial-Services
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run a demo script:**
   ```bash
   python scripts/integration_demo.py
   ```
4. **Explore Jupyter notebooks:**
   ```bash
   jupyter notebook notebooks/eda_preprocessing_demo.ipynb
   ```

---

## Project Structure 

```
Intelligent-Complaint-Analysis-for-Financial-Services/
├── data/                           # Data files
│   ├── complaints.csv              # Raw CFPB complaints data
│   ├── complaints_processed.csv    # Processed dataset (generated)
│   ├── chunked_narratives.csv      # Chunked narratives (generated)
│   └── chunking_summary.csv        # Chunking statistics (generated)
├── notebooks/
│   ├── eda_preprocessing_demo.ipynb # Jupyter notebook with EDA workflow
│   └── text_chunking_demo.ipynb    # Jupyter notebook with chunking workflow
├── src/
│   ├── eda_preprocessing.py        # ComplaintAnalyzer class
│   ├── text_chunking.py            # Text chunking functionality
│   └── embedding_indexing.py       # Embedding and indexing functionality
├── scripts/
│   ├── demo_text_chunking.py       # Text chunking demo script
│   ├── demo_embedding_indexing.py  # Embedding and indexing demo script
│   └── integration_demo.py         # Complete workflow demo script
├── tests/                          # Test files
│   ├── test_constructor.py         # Constructor tests
│   ├── test_text_chunking.py       # Text chunking tests
│   └── test_embedding_indexing.py  # Embedding and indexing tests
├── docs/
│   └── embedding_indexing_report.md # Detailed implementation report
└── README.md                       # This file
```

## Features

### 1. ComplaintAnalyzer Class (EDA & Preprocessing)

A comprehensive Python class that provides:

1. **Data Loading** - Load CFPB complaints CSV with error handling
2. **Exploratory Data Analysis** - Basic dataset exploration and statistics
3. **Data Filtering** - Filter to specific products and remove empty narratives
4. **Text Cleaning** - Clean narratives for better embedding quality
5. **Visualization** - Generate plots and charts for analysis

### 2. Text Chunking Module

Advanced text chunking capabilities for processing long complaint narratives:

1. **Multiple Chunking Strategies** - Recursive character, sentence-based, paragraph-based
2. **Configurable Parameters** - Adjustable chunk size and overlap
3. **Metadata Preservation** - Maintains traceability to original complaints
4. **Performance Analysis** - Comprehensive chunking statistics and optimization

### 3. Embedding and Indexing Module

State-of-the-art embedding and semantic search capabilities:

1. **Embedding Model** - sentence-transformers/all-MiniLM-L6-v2 for optimal performance
2. **Vector Storage** - ChromaDB for efficient similarity search
3. **Metadata Preservation** - Complete traceability for each chunk
4. **Semantic Search** - Fast and accurate similarity search with filtering
5. **Batch Processing** - Efficient handling of large datasets

### Key Methods

#### ComplaintAnalyzer
- `load_complaints_data()` - Load complaints data with error handling
- `initial_eda()` - Perform initial exploratory data analysis
- `plot_product_distribution()` - Visualize complaint distribution by product
- `analyze_narrative_lengths()` - Analyze text length characteristics
- `count_narrative_presence()` - Count narratives vs non-narratives
- `filter_dataset()` - Filter to 5 specified products and remove empty narratives
- `clean_narratives()` - Clean text for better embedding quality

#### Text Chunking
- `chunk_single_narrative()` - Chunk a single narrative text
- `chunk_narratives_batch()` - Process multiple narratives with metadata
- `analyze_performance()` - Analyze chunking performance metrics
- `save_results()` - Save chunked results to various formats

#### Embedding and Indexing
- `generate_embeddings()` - Generate embeddings for text chunks
- `index_chunks()` - Index chunks with metadata in ChromaDB
- `search_similar_chunks()` - Perform semantic similarity search
- `search_complaints_by_topic()` - Topic-based search with filtering
- `get_collection_stats()` - Get vector store statistics

## Workflow

The recommended workflow follows this order:

### Phase 1: EDA and Preprocessing
1. **Data Loading** - Load raw complaints data
2. **Original Analysis** - Analyze raw dataset to understand characteristics
3. **Data Filtering** - Filter to specific products and remove empty narratives
4. **Text Cleaning** - Clean narratives for better embedding quality
5. **Final Analysis** - Analyze the cleaned and filtered dataset

### Phase 2: Text Chunking
6. **Chunking Strategy** - Choose appropriate chunking method and parameters
7. **Batch Processing** - Process narratives into manageable chunks
8. **Performance Analysis** - Analyze chunking effectiveness and optimize

### Phase 3: Embedding and Indexing
9. **Embedding Generation** - Generate vector embeddings using sentence-transformers/all-MiniLM-L6-v2
10. **Vector Storage** - Index embeddings with metadata in ChromaDB
11. **Semantic Search** - Enable similarity search and topic-based retrieval

## Usage

### Using the Jupyter Notebooks

1. **EDA and Preprocessing**: Navigate to `notebooks/` and open `eda_preprocessing_demo.ipynb`
2. **Text Chunking**: Open `text_chunking_demo.ipynb` for chunking workflow
3. Run all cells to execute the complete workflow

### Using the Classes Directly

#### EDA and Preprocessing
```python
from src.eda_preprocessing import ComplaintAnalyzer

# Create analyzer and load data
analyzer = ComplaintAnalyzer('data/complaints.csv')

# Perform analysis
analyzer.initial_eda()
analyzer.plot_product_distribution()

# Filter and clean
analyzer.filter_dataset()
analyzer.clean_narratives()

# Final analysis
analyzer.analyze_narrative_lengths(column_name='Consumer complaint narrative_cleaned')
```

#### Text Chunking
```python
from src.text_chunking import NarrativeChunkingStrategy

# Create chunking strategy
strategy = NarrativeChunkingStrategy(chunk_size=400, chunk_overlap=100)

# Chunk narratives
chunked_narratives = strategy.chunk_narratives_batch(narratives)

# Analyze performance
performance = strategy.analyze_performance(chunked_narratives)
```

#### Embedding and Indexing
```python
from src.embedding_indexing import create_embedding_indexer

# Create indexer
indexer = create_embedding_indexer()

# Index chunks
success = indexer.index_chunks(chunked_narratives)

# Search for similar complaints
results = indexer.search_similar_chunks("fraudulent charges", n_results=10)
```

### Running Demo Scripts

```bash
# Text chunking demo
python scripts/demo_text_chunking.py

# Embedding and indexing demo
python scripts/demo_embedding_indexing.py

# Complete integration demo
python scripts/integration_demo.py
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python tests/test_text_chunking.py
python tests/test_embedding_indexing.py
```

## Data Processing Steps

### 1. Data Filtering
- **Products**: Credit card, Personal loan, Buy Now Pay Later (BNPL), Savings account, Money transfers
- **Narratives**: Remove records with empty Consumer complaint narrative fields

### 2. Text Cleaning
- **Lowercasing**: Convert all text to lowercase
- **Boilerplate Removal**: Remove common complaint boilerplate text
- **Special Characters**: Remove punctuation and special characters
- **Whitespace Normalization**: Clean up extra spaces and formatting
- **Quality Control**: Remove narratives that become too short after cleaning

### 3. Output Generation
- **Visualizations**: Product distributions, narrative length histograms
- **Statistics**: Before/after comparisons, data quality metrics
- **Processed Data**: Clean CSV file ready for NLP tasks

## Dependencies & Installation

- Python 3.8+
- See `requirements.txt` for all dependencies.
- Major packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - sentence-transformers
  - faiss
  - chromadb
  - langchain (optional, for advanced chunking)

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

## Links & Resources

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

Please ensure your code follows the existing style and includes appropriate tests.

## Contact
*For questins, suggestions, or collaboration, feel free to reach out:
- GitHub: [yourusername](https://github.com/wondifraw)
- Email: wondebdu@gmail.com



