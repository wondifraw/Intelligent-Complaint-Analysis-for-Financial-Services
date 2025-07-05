#!/usr/bin/env python3
"""
Test script to verify the constructor fix for DataFrame boolean evaluation.
"""

import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from eda_preprocessing import ComplaintAnalyzer

def test_constructor_with_dataframe():
    """Test the constructor with a DataFrame."""
    
    # Create a sample DataFrame
    data = {
        'Product': ['Credit card', 'Personal loan'],
        'Consumer complaint narrative': [
            'This is a valid complaint narrative.',
            'Another valid narrative here.'
        ]
    }
    
    df = pd.DataFrame(data)
    
    try:
        # This should not raise the boolean evaluation error
        analyzer = ComplaintAnalyzer(df)
        print("✅ Constructor with DataFrame completed successfully!")
        print(f"Analyzer DataFrame shape: {analyzer.df.shape}")
        return True
    except Exception as e:
        print(f"❌ Constructor with DataFrame failed with error: {e}")
        return False

def test_constructor_with_filepath():
    """Test the constructor with a filepath."""
    
    try:
        # This should work if the file exists, or handle the error gracefully
        analyzer = ComplaintAnalyzer("data/complaints.csv")
        print("✅ Constructor with filepath completed successfully!")
        if analyzer.df is not None:
            print(f"Analyzer DataFrame shape: {analyzer.df.shape}")
        else:
            print("No data loaded (file might not exist or be accessible)")
        return True
    except Exception as e:
        print(f"❌ Constructor with filepath failed with error: {e}")
        return False

def test_constructor_with_none():
    """Test the constructor with None."""
    
    try:
        # This should work without error
        analyzer = ComplaintAnalyzer()
        print("✅ Constructor with None completed successfully!")
        print(f"Analyzer DataFrame: {analyzer.df}")
        return True
    except Exception as e:
        print(f"❌ Constructor with None failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Testing constructor fixes...")
    print("=" * 50)
    
    success1 = test_constructor_with_dataframe()
    success2 = test_constructor_with_filepath()
    success3 = test_constructor_with_none()
    
    print("=" * 50)
    if success1 and success2 and success3:
        print("🎉 All constructor tests passed! The boolean evaluation error has been fixed.")
    else:
        print("❌ Some tests failed. Please check the error messages above.") 