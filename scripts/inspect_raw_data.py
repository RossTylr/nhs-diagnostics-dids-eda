#!/usr/bin/env python3
"""
Script to inspect raw DIDS data
This script provides comprehensive analysis of the raw DIDS dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)

def main():
    print("=== DIDS RAW DATA INSPECTION ===\n")
    
    # 1. Load Raw Data
    print("1. LOADING RAW DATA")
    print("=" * 50)
    
    data_path = Path('data/raw/2024_01.csv')
    
    print(f"File exists: {data_path.exists()}")
    if data_path.exists():
        print(f"File size: {data_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Load data
        print("\nLoading data...")
        df = pd.read_csv(data_path, low_memory=False)
        print(f"Data loaded successfully!")
        print(f"Shape: {df.shape}")
        
        # 2. Basic Data Overview
        print("\n\n2. BASIC DATA OVERVIEW")
        print("=" * 50)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Number of rows: {df.shape[0]:,}")
        print(f"Number of columns: {df.shape[1]}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
        
        print("\nFirst few rows:")
        print(df.head())
        
        # 3. Column Information
        print("\n\n3. COLUMN INFORMATION")
        print("=" * 50)
        
        print("\nColumn names (indexed):")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        
        # Create column info DataFrame
        column_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        
        print(f"\nDetailed column info:")
        print(column_info)
        
        # 4. Missing Values Analysis
        print("\n\n4. MISSING VALUES ANALYSIS")
        print("=" * 50)
        
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df) * 100).round(2)
        
        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        }).sort_values('Missing Percentage', ascending=False)
        
        print(f"Total missing values: {missing_data.sum():,}")
        print(f"Columns with missing values: {(missing_data > 0).sum()}")
        
        print("\nColumns with missing values:")
        print(missing_summary[missing_summary['Missing Count'] > 0])
        
        # 5. Data Type Analysis
        print("\n\n5. DATA TYPE ANALYSIS")
        print("=" * 50)
        
        # Identify date columns
        date_columns = []
        for col in df.columns[:5]:  # First 5 columns appear to be dates
            try:
                pd.to_datetime(df[col].iloc[0])
                date_columns.append(col)
            except:
                pass
        
        print(f"Potential date columns: {date_columns}")
        
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nNumeric columns: {numeric_columns}")
        
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        print(f"\nCategorical columns: {categorical_columns}")
        
        # 6. Column-by-Column Analysis (first 10 columns)
        print("\n\n6. DETAILED COLUMN ANALYSIS")
        print("=" * 50)
        
        for col in df.columns[:10]:
            analyze_column(df, col)
        
        # 7. Data Quality Assessment
        print("\n\n7. DATA QUALITY ASSESSMENT")
        print("=" * 50)
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        print(f"Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
        
        # Check for empty strings
        empty_strings = (df == '').sum().sum()
        print(f"Empty strings: {empty_strings:,}")
        
        # Check for 'NULL' strings
        null_strings = (df == 'NULL').sum().sum()
        print(f"'NULL' strings: {null_strings:,}")
        
        # Summary statistics for numeric columns
        print(f"\nNumeric columns summary:")
        numeric_summary = df.select_dtypes(include=[np.number]).describe()
        print(numeric_summary)
        
        # 8. Save Results
        print("\n\n8. SAVING RESULTS")
        print("=" * 50)
        
        # Create outputs directory if it doesn't exist
        outputs_dir = Path('outputs')
        outputs_dir.mkdir(exist_ok=True)
        
        # Save analysis results
        column_info.to_csv('outputs/column_analysis.csv', index=False)
        missing_summary.to_csv('outputs/missing_values_analysis.csv', index=False)
        
        print("Analysis results saved to outputs/ directory")
        
        # Create summary report
        summary_report = {
            'Total Rows': len(df),
            'Total Columns': len(df.columns),
            'Memory Usage (MB)': df.memory_usage(deep=True).sum() / (1024*1024),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicate Rows': df.duplicated().sum(),
            'Numeric Columns': len(df.select_dtypes(include=[np.number]).columns),
            'Categorical Columns': len(df.select_dtypes(include=['object']).columns),
            'Date Columns': len(date_columns)
        }
        
        print("\n=== SUMMARY REPORT ===")
        for key, value in summary_report.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:,}")
    
    else:
        print("Error: Data file not found!")

def analyze_column(df, column_name):
    """Analyze a single column and return summary statistics"""
    col_data = df[column_name]
    
    print(f"\n{'='*50}")
    print(f"ANALYSIS FOR COLUMN: {column_name}")
    print(f"{'='*50}")
    
    # Basic info
    print(f"Data type: {col_data.dtype}")
    print(f"Total values: {len(col_data):,}")
    print(f"Non-null values: {col_data.count():,}")
    print(f"Null values: {col_data.isnull().sum():,}")
    print(f"Unique values: {col_data.nunique():,}")
    
    # Sample values
    print(f"\nSample values:")
    sample_values = col_data.dropna().unique()[:10]
    for i, val in enumerate(sample_values, 1):
        print(f"  {i}. {val}")
    
    # Value counts for categorical data
    if col_data.dtype == 'object' and col_data.nunique() < 50:
        print(f"\nValue counts (top 10):")
        value_counts = col_data.value_counts().head(10)
        for val, count in value_counts.items():
            print(f"  {val}: {count:,} ({count/len(col_data)*100:.1f}%)")
    
    # Numeric statistics
    if pd.api.types.is_numeric_dtype(col_data):
        print(f"\nNumeric statistics:")
        print(f"  Min: {col_data.min()}")
        print(f"  Max: {col_data.max()}")
        print(f"  Mean: {col_data.mean():.2f}")
        print(f"  Median: {col_data.median():.2f}")
        print(f"  Std: {col_data.std():.2f}")
    
    # Date statistics
    if pd.api.types.is_datetime64_any_dtype(col_data):
        print(f"\nDate statistics:")
        print(f"  Min date: {col_data.min()}")
        print(f"  Max date: {col_data.max()}")
        print(f"  Date range: {col_data.max() - col_data.min()}")

if __name__ == "__main__":
    main() 