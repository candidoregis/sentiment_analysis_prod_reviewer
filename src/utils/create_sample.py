import pandas as pd
import numpy as np

# Read the full dataset
try:
    print("Reading full dataset...")
    df = pd.read_csv('Reviews.csv', encoding='utf-8')
    print(f"Full dataset size: {len(df)} rows")
    
    # Create a stratified sample to maintain sentiment distribution
    sample_size = 10000  # Adjust this number as needed
    df_sample = df.groupby('Score').apply(lambda x: x.sample(min(len(x), sample_size//5), random_state=42))
    df_sample = df_sample.reset_index(drop=True)
    
    print(f"\nSample dataset size: {len(df_sample)} rows")
    print("\nOriginal sentiment distribution:")
    print(df['Score'].value_counts())
    print("\nSample sentiment distribution:")
    print(df_sample['Score'].value_counts())
    
    # Save the sample dataset
    df_sample.to_csv('Reviews_sample.csv', index=False)
    print("\nSample dataset saved successfully!")
    
except Exception as e:
    print(f"Error: {str(e)}")
