#!/usr/bin/env python3
"""
SerpApi Integration

This module provides functions to search for products using SerpApi based on product titles.
"""

import os
import json
from dotenv import load_dotenv
from serpapi import GoogleSearch
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Get the absolute path to the config directory
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'config'))
DEFAULT_CREDENTIALS_FILE = os.path.join(CONFIG_DIR, "credentials.json")

def get_serp_api_key(credentials_file=DEFAULT_CREDENTIALS_FILE):
    """
    Get SerpApi key from environment variables or credentials file.
    Environment variables take precedence over the credentials file.
    
    Args:
        credentials_file (str): Path to the credentials file
        
    Returns:
        str: SerpApi key or None if not found
    """
    # First try to get the key from environment variables
    api_key = os.environ.get("SERPAPI_KEY")
    if api_key:
        return api_key
    
    # If not found in environment variables, try the credentials file
    try:
        # Check if credentials file exists
        if not os.path.exists(credentials_file):
            print(f"Credentials file '{credentials_file}' not found and no environment variables set.")
            return None
        
        # Load credentials from file
        with open(credentials_file, 'r') as f:
            credentials = json.load(f)
        
        # Get SerpApi key from credentials
        if "serpapi" in credentials and "key" in credentials["serpapi"]:
            api_key = credentials["serpapi"]["key"]
            
            # Check if the key is a placeholder
            if api_key == "your_serpapi_key":
                print(f"Please update your SerpApi key in '{credentials_file}' or set the SERPAPI_KEY environment variable.")
                return None
                
            return api_key
        else:
            print(f"No SerpApi key found in '{credentials_file}' or environment variables.")
            return None
            
    except Exception as e:
        print(f"Error loading SerpApi key: {e}")
        return None

def search_products(product_title, max_results=5):
    """
    Search for products using SerpApi based on the product title.
    
    Args:
        product_title (str): The title of the product to search for
        max_results (int): Maximum number of results to return
        
    Returns:
        list: List of dictionaries containing product information
    """
    api_key = get_serp_api_key()
    if not api_key:
        return []
    
    try:
        # Prepare search parameters
        params = {
            "engine": "google_shopping",
            "q": product_title,
            "api_key": api_key,
            "gl": "us",  # Country code (United States)
            "hl": "en"   # Language (English)
        }
        
        # Execute search
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Check if shopping_results exist
        if "shopping_results" not in results:
            print(f"No shopping results found for '{product_title}'")
            return []
        
        # Extract relevant information from results
        products = []
        for item in results["shopping_results"][:max_results]:
            product = {
                "title": item.get("title", ""),
                "price": item.get("price", ""),
                "link": item.get("link", ""),
                "source": item.get("source", ""),
                "rating": item.get("rating", ""),
                "reviews": item.get("reviews", ""),
                "thumbnail": item.get("thumbnail", "")
            }
            products.append(product)
        
        return products
    
    except Exception as e:
        print(f"Error searching products with SerpApi: {str(e)}")
        return []

def get_product_alternatives(product_title, max_results=5, max_words=7):
    """
    Get alternative products based on the product title.
    This is the main function to be used by other modules.
    
    Args:
        product_title (str): The title of the product to search for
        max_results (int): Maximum number of results to return
        max_words (int): Maximum number of words to include in the search query
        
    Returns:
        list: List of dictionaries containing alternative product information
    """
    # Clean up the product title for better search results and limit to max_words
    clean_title = clean_product_title(product_title, max_words)
    
    # Search for products
    alternatives = search_products(clean_title, max_results)
    
    # If no results, try with a simplified title
    if not alternatives and len(clean_title.split()) > 2:
        # Try with just the first two words of the title
        simplified_title = " ".join(clean_title.split()[:2])
        print(f"No results found. Trying with simplified title: '{simplified_title}'")
        alternatives = search_products(simplified_title, max_results)
    
    # Ensure we always return a list, even if empty
    if alternatives is None:
        alternatives = []
        
    return alternatives

def clean_product_title(title, max_words=7):
    """
    Clean up a product title for better search results and limit to a specified number of words.
    Removes unnecessary information like model numbers, specific details, etc.
    
    Args:
        title (str): The original product title
        max_words (int): Maximum number of words to include in the search query
        
    Returns:
        str: Cleaned and truncated product title
    """
    import re
    
    # If title is empty or None, return a default value
    if not title:
        return "product"
    
    # Remove text in parentheses, brackets, or after specific characters
    title = re.sub(r'\([^)]*\)', '', title)  # Remove text in parentheses
    title = re.sub(r'\[[^\]]*\]', '', title)  # Remove text in brackets
    
    # Split by common separators and take the first part
    parts = re.split(r'[,;:\-|]', title)
    if parts:
        title = parts[0].strip()
    
    # Remove multiple spaces
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Limit to max_words
    words = title.split()
    if len(words) > max_words:
        title = ' '.join(words[:max_words])
        print(f"Title truncated to {max_words} words: '{title}'")
    
    return title

def _test():
    """
    Test function to demonstrate usage.
    """
    # Test with a sample product title
    product_title = "Apple iPad Pro 12.9-inch (6th Generation): with M2 chip, Liquid Retina XDR Display"
    print(f"Searching for alternatives to: {product_title}")
    
    alternatives = get_product_alternatives(product_title)
    
    if alternatives:
        print(f"\nFound {len(alternatives)} alternatives:")
        for i, product in enumerate(alternatives, 1):
            print(f"\n{i}. {product['title']}")
            print(f"   Price: {product['price']}")
            print(f"   Source: {product['source']}")
            print(f"   Rating: {product['rating']} ({product['reviews']} reviews)")
            print(f"   Link: {product['link']}")
    else:
        print("No alternatives found.")

if __name__ == "__main__":
    _test()
