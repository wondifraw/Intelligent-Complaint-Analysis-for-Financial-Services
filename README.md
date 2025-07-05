# Intelligent Complaint Analysis for Financial Services

This project implements a comprehensive EDA (Exploratory Data Analysis) and preprocessing pipeline for CFPB (Consumer Financial Protection Bureau) complaints data.

## Project Structure 

```
Intelligent-Complaint-Analysis-for-Financial-Services/
├── data/                           # Data files
│   ├── complaints.csv              # Raw CFPB complaints data
│   └── complaints_processed.csv    # Processed dataset (generated)
├── notebooks/
│   └── eda_preprocessing_demo.ipynb # Jupyter notebook with complete workflow
├── src/
│   └── eda_preprocessing.py        # ComplaintAnalyzer class
├── tests/                          # Test files
├── test_analyzer.py                # Test script for the analyzer
└── README.md                       # This file
```

## Features

### ComplaintAnalyzer Class

A comprehensive Python class that provides:

1. **Data Loading** - Load CFPB complaints CSV with error handling
2. **Exploratory Data Analysis** - Basic dataset exploration and statistics
3. **Data Filtering** - Filter to specific products and remove empty narratives
4. **Text Cleaning** - Clean narratives for better embedding quality
5. **Visualization** - Generate plots and charts for analysis

### Key Methods

- `load_complaints_data()` - Load complaints data with error handling
- `initial_eda()` - Perform initial exploratory data analysis
- `plot_product_distribution()` - Visualize complaint distribution by product
- `analyze_narrative_lengths()` - Analyze text length characteristics
- `count_narrative_presence()` - Count narratives vs non-narratives
- `filter_dataset()` - Filter to 5 specified products and remove empty narratives
- `clean_narratives()` - Clean text for better embedding quality

## Workflow

The recommended workflow follows this order:

1. **Data Loading** - Load raw complaints data
2. **Original Analysis** - Analyze raw dataset to understand characteristics
3. **Data Filtering** - Filter to specific products and remove empty narratives
4. **Text Cleaning** - Clean narratives for better embedding quality
5. **Final Analysis** - Analyze the cleaned and filtered dataset

## Usage

### Using the Jupyter Notebook

1. Navigate to the `notebooks/` directory
2. Open `eda_preprocessing_demo.ipynb`
3. Run all cells to execute the complete workflow

### Using the Class Directly

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

### Running Tests

```bash
python test_analyzer.py
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

## Requirements

- Python 3.7+
- pandas
- matplotlib
- seaborn
- numpy

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install pandas matplotlib seaborn numpy
   ```
3. Place your CFPB complaints data in `data/complaints.csv`

## Output Files

The workflow generates several output files:

- `product_distribution_original.png` - Original product distribution
- `narrative_lengths_original.png` - Original narrative lengths
- `product_distribution_final.png` - Final product distribution
- `narrative_lengths_final.png` - Final narrative lengths
- `narrative_lengths_comparison.png` - Before vs after comparison
- `complaints_processed.csv` - Final processed dataset

## Next Steps

The processed data is now ready for:
- Text embedding generation
- Sentiment analysis
- Topic modeling
- Complaint classification
- Other NLP tasks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.