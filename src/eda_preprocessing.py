import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string


class ComplaintAnalyzer:
    """
    A class for performing exploratory data analysis and preprocessing on CFPB complaints data.
    """
    
    def __init__(self, filepath_or_df=None):
        """
        Initialize the ComplaintAnalyzer.
        Args:
            filepath_or_df (str or pd.DataFrame, optional): Path to the complaints CSV file or DataFrame.
        """
        self.filepath = None
        self.df = None
        
        if filepath_or_df is not None:
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
    
    def load_complaints_data(self, filepath=None):
        """
        Loads the CFPB complaints CSV dataset with error handling.
        Args:
            filepath (str): Path to the complaints CSV file.
        Returns:
            pd.DataFrame: Loaded DataFrame, or None if error occurs.
        """
        if filepath is None:
            filepath = self.filepath
            
        try:
            # Use chunking for large files if needed, but here we try to load all
            self.df = pd.read_csv(filepath, low_memory=False)
            print(f"Loaded data with shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except pd.errors.EmptyDataError:
            print("No data: file is empty.")
        except Exception as e:
            print(f"Error loading data: {e}")
        return None

    def initial_eda(self, df=None):
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
            
        print("\n--- Data Head ---")
        print(df.head())
        print("\n--- Data Info ---")
        print(df.info())
        print("\n--- Missing Values ---")
        print(df.isnull().sum())
        print("\n--- Basic Statistics ---")
        print(df.describe(include='all'))

    def plot_product_distribution(self, df=None, save_path=None):
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
            
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, y='Product', order=df['Product'].value_counts().index)
        plt.title('Distribution of Complaints by Product')
        plt.xlabel('Number of Complaints')
        plt.ylabel('Product')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def analyze_narrative_lengths(self, df=None, save_path=None, column_name='Consumer complaint narrative'):
        """
        Calculates and visualizes the length (word count) of Consumer complaint narratives.
        Args:
            df (pd.DataFrame, optional): The complaints DataFrame. If None, uses self.df.
            save_path (str, optional): If provided, saves the plot to this path.
            column_name (str): Name of the column containing narratives.
        Returns:
            pd.Series: Series of word counts for narratives.
        """
        if df is None:
            df = self.df
            
        if df is None:
            print("No data loaded. Please load data first.")
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
            plt.savefig(save_path)
        plt.show()
        return word_counts

    def count_narrative_presence(self, df=None):
        """
        Counts the number of complaints with and without narratives.
        Args:
            df (pd.DataFrame, optional): The complaints DataFrame. If None, uses self.df.
        Returns:
            tuple: (with_narrative, without_narrative)
        """
        if df is None:
            df = self.df
            
        if df is None:
            print("No data loaded. Please load data first.")
            return None, None
            
        with_narrative = df['Consumer complaint narrative'].notnull().sum()
        without_narrative = df['Consumer complaint narrative'].isnull().sum()
        print(f"\nComplaints with narrative: {with_narrative}")
        print(f"Complaints without narrative: {without_narrative}")
        return with_narrative, without_narrative
    
    def get_dataframe(self):
        """
        Returns the loaded DataFrame.
        Returns:
            pd.DataFrame: The loaded complaints DataFrame.
        """
        return self.df
    
    def set_dataframe(self, df):
        """
        Sets the DataFrame for analysis.
        Args:
            df (pd.DataFrame): The complaints DataFrame to analyze.
        """
        self.df = df
        print(f"DataFrame set with shape: {df.shape}")

    def filter_dataset(self, df=None):
        """
        Filters the dataset to meet project requirements:
        - Include only records for the five specified products: Credit card, Personal loan, 
          Buy Now, Pay Later (BNPL), Savings account, Money transfers
        - Remove any records with empty Consumer complaint narrative fields
        
        Args:
            df (pd.DataFrame, optional): The complaints DataFrame. If None, uses self.df.
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if df is None:
            df = self.df
            
        if df is None:
            print("No data loaded. Please load data first.")
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

    def clean_narratives(self, df=None, column_name='Consumer complaint narrative'):
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
            pd.DataFrame: DataFrame with cleaned narratives
        """
        if df is None:
            df = self.df
            
        if df is None:
            print("No data loaded. Please load data first.")
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
        
        # Apply cleaning to the narrative column
        df_cleaned[f'{column_name}_cleaned'] = df_cleaned[column_name].apply(clean_text)
        
        # Remove rows where cleaned narrative is empty or too short (less than 10 characters)
        # Handle potential NaN values before using str methods
        df_cleaned = df_cleaned[df_cleaned[f'{column_name}_cleaned'].fillna('').str.len() >= 10].copy()
        
        print(f"After cleaning: {df_cleaned.shape[0]} records")
        
        # Show some examples of before and after cleaning
        print("\n--- Text Cleaning Examples ---")
        for i in range(min(3, len(df_cleaned))):
            original = df_cleaned[column_name].iloc[i][:200] + "..." if len(df_cleaned[column_name].iloc[i]) > 200 else df_cleaned[column_name].iloc[i]
            cleaned = df_cleaned[f'{column_name}_cleaned'].iloc[i][:200] + "..." if len(df_cleaned[f'{column_name}_cleaned'].iloc[i]) > 200 else df_cleaned[f'{column_name}_cleaned'].iloc[i]
            print(f"\nExample {i+1}:")
            print(f"Original: {original}")
            print(f"Cleaned:  {cleaned}")
        
        # Update the instance DataFrame
        self.df = df_cleaned
        print(f"\nFinal cleaned dataset shape: {self.df.shape}")
        
        return self.df


# Example usage and testing
if __name__ == "__main__":
    # This file contains the ComplaintAnalyzer class
    # For demonstration and testing, use the notebook: notebooks/eda_preprocessing_demo.ipynb
    print("ComplaintAnalyzer class loaded successfully!")
    print("For demonstration and testing, please use the notebook: notebooks/eda_preprocessing_demo.ipynb")


