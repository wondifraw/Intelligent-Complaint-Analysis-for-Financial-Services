import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string


class ComplaintAnalyzer:
    """
    A class for performing exploratory data analysis and preprocessing on CFPB complaints data.

    Example:
        >>> analyzer = ComplaintAnalyzer('data/complaints.csv')
        >>> analyzer.initial_eda()
        >>> analyzer.filter_dataset()
        >>> analyzer.clean_narratives()
    """
    
    def __init__(self, filepath_or_df: 'str | pd.DataFrame' = None):
        """
        Initialize the ComplaintAnalyzer.

        Args:
            filepath_or_df (str or pd.DataFrame, optional): Path to the complaints CSV file or a DataFrame.
        """
        self.filepath = None
        self.df = None
        
        if filepath_or_df is not None:
            try:
                if isinstance(filepath_or_df, str):
                    # If it's a string, treat it as a filepath
                    self.filepath = filepath_or_df
                    self.load_complaints_data(filepath_or_df)
                elif hasattr(filepath_or_df, 'shape'):  # Check if it's a DataFrame-like object
                    # If it's a DataFrame, set it directly
                    self.df = filepath_or_df
                    print(f"DataFrame set with shape: {filepath_or_df.shape}")
                else:
                    print(f"Warning: Unsupported type {type(filepath_or_df)}. Please provide a filepath (str) or DataFrame.")
            except Exception as e:
                print(f"Error during initialization: {e}")
    
    def load_complaints_data(self, filepath: str = None, chunk_size: int = 100000) -> 'pd.DataFrame | None':
        """
        Loads the CFPB complaints CSV dataset with chunked reading to handle large files.

        Args:
            filepath (str, optional): Path to the complaints CSV file. If None, uses self.filepath.
            chunk_size (int): Number of rows to read at a time. Adjust based on available memory.

        Returns:
            pd.DataFrame or None: Loaded DataFrame, or None if error occurs.

        Raises:
            FileNotFoundError: If the file is not found.
            pd.errors.EmptyDataError: If the file is empty.
            Exception: For other errors during loading.
        """
        if filepath is None:
            filepath = self.filepath
            
        try:
            print(f"Loading data in chunks of {chunk_size} rows...")
            
            # Read the CSV file in chunks to manage memory usage
            chunks = []
            for chunk in pd.read_csv(filepath, low_memory=False, chunksize=chunk_size):
                # Optionally process each chunk here (e.g., filter, clean, etc.)
                chunks.append(chunk)
            
            # Concatenate all chunks into a single DataFrame
            self.df = pd.concat(chunks, ignore_index=True)
            print(f"Loaded data with shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except pd.errors.EmptyDataError:
            print("No data: file is empty.")
        except Exception as e:
            print(f"Error loading data: {e}")
        return None

    def initial_eda(self, df: 'pd.DataFrame | None' = None) -> None:
        """
        Prints initial EDA: head, info, missing values, and basic stats.

        Args:
            df (pd.DataFrame, optional): The complaints DataFrame. If None, uses self.df.
        """
        if df is None:
            df = self.df
            
        if df is None:
            print("No data loaded. Please load data first.")
            return
            
        try:
            print("\n--- Data Head ---")
            print(df.head())
            print("\n--- Data Info ---")
            print(df.info())
            print("\n--- Missing Values ---")
            print(df.isnull().sum())
            print("\n--- Basic Statistics ---")
            print(df.describe(include='all'))
        except Exception as e:
            print(f"Error during initial EDA: {e}")

    def plot_product_distribution(self, df: 'pd.DataFrame | None' = None, save_path: str = None) -> None:
        """
        Plots and shows the distribution of complaints across Products.

        Args:
            df (pd.DataFrame, optional): The complaints DataFrame. If None, uses self.df.
            save_path (str, optional): If provided, saves the plot to this path.
        """
        if df is None:
            df = self.df
            
        if df is None:
            print("No data loaded. Please load data first.")
            return
            
        try:
            plt.figure(figsize=(12, 6))
            sns.countplot(data=df, y='Product', order=df['Product'].value_counts().index)
            plt.title('Distribution of Complaints by Product')
            plt.xlabel('Number of Complaints')
            plt.ylabel('Product')
            plt.tight_layout()
            if save_path:
                try:
                    plt.savefig(save_path)
                except Exception as e:
                    print(f"Error saving plot to {save_path}: {e}")
            plt.show()
        except Exception as e:
            print(f"Error creating product distribution plot: {e}")

    def analyze_narrative_lengths(self, df: 'pd.DataFrame | None' = None, save_path: str = None, column_name: str = 'Consumer complaint narrative') -> 'pd.Series | None':
        """
        Calculates and visualizes the length (word count) of Consumer complaint narratives.

        Args:
            df (pd.DataFrame, optional): The complaints DataFrame. If None, uses self.df.
            save_path (str, optional): If provided, saves the plot to this path.
            column_name (str): Name of the column containing narratives.

        Returns:
            pd.Series or None: Series of word counts for narratives, or None if error occurs.
        """
        if df is None:
            df = self.df
            
        if df is None:
            print("No data loaded. Please load data first.")
            return None
            
        try:
            # Check if column exists
            if column_name not in df.columns:
                print(f"Column '{column_name}' not found in DataFrame. Available columns: {list(df.columns)}")
                return None
                
            # Fill NaN with empty string for word count
            narratives = df[column_name].fillna("")
            word_counts = narratives.apply(lambda x: len(str(x).split()))
            print(f"\n--- Narrative Lengths ({column_name}) ---\nMin: {word_counts.min()}, Max: {word_counts.max()}, Mean: {word_counts.mean():.2f}")
            
            plt.figure(figsize=(10, 5))
            sns.histplot(word_counts, bins=50, kde=True)
            plt.title(f'Distribution of Narrative Word Counts - {column_name}')
            plt.xlabel('Word Count')
            plt.ylabel('Number of Complaints')
            plt.tight_layout()
            if save_path:
                try:
                    plt.savefig(save_path)
                except Exception as e:
                    print(f"Error saving plot to {save_path}: {e}")
            plt.show()
            return word_counts
        except Exception as e:
            print(f"Error analyzing narrative lengths: {e}")
            return None

    def count_narrative_presence(self, df: 'pd.DataFrame | None' = None) -> 'tuple[int | None, int | None]':
        """
        Counts the number of complaints with and without narratives.

        Args:
            df (pd.DataFrame, optional): The complaints DataFrame. If None, uses self.df.

        Returns:
            tuple: (with_narrative, without_narrative) or (None, None) if error occurs.
        """
        if df is None:
            df = self.df
            
        if df is None:
            print("No data loaded. Please load data first.")
            return None, None
            
        try:
            # Check if column exists
            if 'Consumer complaint narrative' not in df.columns:
                print("Column 'Consumer complaint narrative' not found in DataFrame.")
                return None, None
                
            with_narrative = df['Consumer complaint narrative'].notnull().sum()
            without_narrative = df['Consumer complaint narrative'].isnull().sum()
            print(f"\nComplaints with narrative: {with_narrative}")
            print(f"Complaints without narrative: {without_narrative}")
            return with_narrative, without_narrative
        except Exception as e:
            print(f"Error counting narrative presence: {e}")
            return None, None
    
    def get_dataframe(self) -> 'pd.DataFrame | None':
        """
        Returns the loaded DataFrame.

        Returns:
            pd.DataFrame or None: The loaded complaints DataFrame, or None if not loaded.
        """
        return self.df
    
    def set_dataframe(self, df: 'pd.DataFrame') -> None:
        """
        Sets the DataFrame for analysis.

        Args:
            df (pd.DataFrame): The complaints DataFrame to analyze.
        """
        try:
            self.df = df
            print(f"DataFrame set with shape: {df.shape}")
        except Exception as e:
            print(f"Error setting DataFrame: {e}")

    def filter_dataset(self, df: 'pd.DataFrame | None' = None) -> 'pd.DataFrame | None':
        """
        Filters the dataset to meet project requirements:
        - Include only records for the five specified products: Credit card, Personal loan, 
          Buy Now, Pay Later (BNPL), Savings account, Money transfers
        - Remove any records with empty Consumer complaint narrative fields
        
        Args:
            df (pd.DataFrame, optional): The complaints DataFrame. If None, uses self.df.
        Returns:
            pd.DataFrame or None: Filtered DataFrame, or None if error occurs.
        """
        if df is None:
            df = self.df
            
        if df is None:
            print("No data loaded. Please load data first.")
            return None
            
        try:
            # Check if required columns exist
            required_columns = ['Product', 'Consumer complaint narrative']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                return None
                
            print(f"Original dataset shape: {df.shape}")
            
            # Define the five specified products
            target_products = [
                'Credit card',
                'Personal loan', 
                'Buy Now, Pay Later (BNPL)',
                'Savings account',
                'Money transfers'
            ]
            
            # Filter by specified products
            df_filtered = df[df['Product'].isin(target_products)].copy()
            print(f"After filtering by products: {df_filtered.shape[0]} records")
            
            # Remove records with empty Consumer complaint narrative fields
            df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].notna()].copy()
            # Handle potential NaN values before using str methods
            df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].fillna('').str.strip() != ''].copy()
            print(f"After removing empty narratives: {df_filtered.shape[0]} records")
            
            # Show product distribution after filtering
            print("\n--- Product Distribution After Filtering ---")
            product_counts = df_filtered['Product'].value_counts()
            for product, count in product_counts.items():
                print(f"{product}: {count} complaints")
            
            # Update the instance DataFrame
            self.df = df_filtered
            print(f"\nFinal filtered dataset shape: {self.df.shape}")
            
            return self.df
        except Exception as e:
            print(f"Error filtering dataset: {e}")
            return None

    def clean_narratives(self, df: 'pd.DataFrame | None' = None, column_name: str = 'Consumer complaint narrative') -> 'pd.DataFrame | None':
        """
        Cleans the text narratives to improve embedding quality.
        This includes:
        - Lowercasing text
        - Removing special characters and punctuation
        - Removing boilerplate text patterns
        - Normalizing whitespace
        - Removing extra spaces
        
        Args:
            df (pd.DataFrame, optional): The complaints DataFrame. If None, uses self.df.
            column_name (str): Name of the column containing narratives.
        Returns:
            pd.DataFrame or None: DataFrame with cleaned narratives, or None if error occurs.
        """
        if df is None:
            df = self.df
            
        if df is None:
            print("No data loaded. Please load data first.")
            return None
            
        try:
            # Check if column exists
            if column_name not in df.columns:
                print(f"Column '{column_name}' not found in DataFrame. Available columns: {list(df.columns)}")
                return None
                
            print(f"Cleaning narratives in column: {column_name}")
            print(f"Original dataset shape: {df.shape}")
            
            # Create a copy to avoid modifying the original
            df_cleaned = df.copy()
            
            # Common boilerplate patterns to remove
            boilerplate_patterns = [
                r'i am writing to file a complaint',
                r'i am filing this complaint',
                r'this is a complaint about',
                r'i would like to file a complaint',
                r'please investigate this complaint',
                r'this complaint is regarding',
                r'i am submitting this complaint',
                r'formal complaint',
                r'consumer complaint',
                r'cfpb complaint',
                r'complaint number',
                r'reference number',
                r'case number',
                r'account number',
                r'customer service',
                r'customer support',
                r'please help',
                r'thank you',
                r'regards',
                r'sincerely',
                r'best regards',
                r'yours truly',
                r'respectfully',
                r'cc:',
                r'cc :',
                r'to:',
                r'from:',
                r'subject:',
                r'date:',
                r're:',
                r'fw:',
                r'fwd:'
            ]
            
            def clean_text(text):
                try:
                    if pd.isna(text) or text == '':
                        return text
                        
                    # Convert to string if not already
                    text = str(text)
                    
                    # Lowercase
                    text = text.lower()
                    
                    # Remove boilerplate patterns
                    for pattern in boilerplate_patterns:
                        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                    
                    # Remove special characters and punctuation (keep apostrophes for contractions)
                    text = re.sub(r'[^\w\s\']', ' ', text)
                    
                    # Remove extra whitespace
                    text = re.sub(r'\s+', ' ', text)
                    
                    # Remove leading/trailing whitespace
                    text = text.strip()
                    
                    # Remove single characters (likely noise)
                    text = re.sub(r'\b\w\b', '', text)
                    
                    # Clean up extra spaces again
                    text = re.sub(r'\s+', ' ', text)
                    text = text.strip()
                    
                    return text
                except Exception as e:
                    print(f"Error cleaning text: {e}")
                    return text
            
            # Apply cleaning to the narrative column
            df_cleaned[f'{column_name}_cleaned'] = df_cleaned[column_name].apply(clean_text)
            
            # Remove rows where cleaned narrative is empty or too short (less than 10 characters)
            # Handle potential NaN values before using str methods
            df_cleaned = df_cleaned[df_cleaned[f'{column_name}_cleaned'].fillna('').str.len() >= 10].copy()
            
            print(f"After cleaning: {df_cleaned.shape[0]} records")
            
            # Show some examples of before and after cleaning
            print("\n--- Text Cleaning Examples ---")
            for i in range(min(3, len(df_cleaned))):
                try:
                    original = df_cleaned[column_name].iloc[i][:200] + "..." if len(df_cleaned[column_name].iloc[i]) > 200 else df_cleaned[column_name].iloc[i]
                    cleaned = df_cleaned[f'{column_name}_cleaned'].iloc[i][:200] + "..." if len(df_cleaned[f'{column_name}_cleaned'].iloc[i]) > 200 else df_cleaned[f'{column_name}_cleaned'].iloc[i]
                    print(f"\nExample {i+1}:")
                    print(f"Original: {original}")
                    print(f"Cleaned:  {cleaned}")
                except Exception as e:
                    print(f"Error showing example {i+1}: {e}")
            
            # Update the instance DataFrame
            self.df = df_cleaned
            print(f"\nFinal cleaned dataset shape: {self.df.shape}")
            
            return self.df
        except Exception as e:
            print(f"Error cleaning narratives: {e}")
            return None



