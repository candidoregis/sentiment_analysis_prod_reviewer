#!/usr/bin/env python3
"""
Amazon Review Scraper and Sentiment Analysis Tool
Main entry point for the application
"""

import os
import sys
import subprocess

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """
    Run the Streamlit web application
    """
    # Get the absolute path to the app.py file
    app_path = os.path.join(os.path.dirname(__file__), 'src', 'web', 'app.py')
    
    # Run the Streamlit app
    print("Starting E-commerce Product Analyzer...")
    subprocess.run(['streamlit', 'run', app_path])

if __name__ == "__main__":
    main()
