import pandas as pd
import sys

print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")

try:
    print("\nReading CSV file...")
    df = pd.read_csv('Reviews.csv', encoding='utf-8')
    print(f"\nSuccessfully read CSV with {len(df)} rows")
    print("\nFirst few rows:")
    print(df.head(2))
    print("\nColumn types:")
    print(df.dtypes)
    
    # Check for any null values
    print("\nNull values per column:")
    print(df.isnull().sum())
    
except Exception as e:
    print("\nError reading CSV:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    print("\nTraceback:")
    print(traceback.format_exc())
