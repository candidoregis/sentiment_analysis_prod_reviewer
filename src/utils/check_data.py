import pandas as pd

# Load the CSV file
try:
    df = pd.read_csv('Reviews.csv')
    print("\nDataset Columns:")
    print(df.columns.tolist())
    
    print("\nFirst few rows:")
    print(df.head(2))
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
except Exception as e:
    print(f"Error reading CSV: {str(e)}")
