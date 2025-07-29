#!/usr/bin/env python3
"""
Credentials Manager

This module provides functions to load and save credentials from environment variables or a JSON file.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the absolute path to the config directory
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'config'))
DEFAULT_CREDENTIALS_FILE = os.path.join(CONFIG_DIR, "credentials.json")

def load_credentials(site="amazon", credentials_file=DEFAULT_CREDENTIALS_FILE):
    """
    Load credentials for a specific site from environment variables first, then from the credentials file.
    
    Args:
        site (str): The site to load credentials for (default: "amazon")
        credentials_file (str): Path to the credentials file
        
    Returns:
        tuple: (email, password) if found, (None, None) otherwise
    """
    try:
        # First, try to get credentials from environment variables
        env_email_key = f"{site.upper()}_EMAIL"
        env_password_key = f"{site.upper()}_PASSWORD"
        
        email = os.environ.get(env_email_key)
        password = os.environ.get(env_password_key)
        
        # If both email and password are found in environment variables, use them
        if email and password:
            return email, password
            
        # Otherwise, fall back to credentials file
        # Check if credentials file exists
        if not os.path.exists(credentials_file):
            print(f"Credentials file '{credentials_file}' not found and no environment variables set.")
            return None, None
        
        # Load credentials from file
        with open(credentials_file, 'r') as f:
            credentials = json.load(f)
        
        # Get credentials for the specified site
        if site in credentials:
            site_credentials = credentials[site]
            email = site_credentials.get('email')
            password = site_credentials.get('password')
            
            # Check if credentials are placeholders
            if email == "your_email@example.com" or password == "your_password":
                print(f"Please update your {site} credentials in '{credentials_file}' or set environment variables")
                return None, None
                
            return email, password
        else:
            print(f"No credentials found for '{site}' in '{credentials_file}' or environment variables")
            return None, None
            
    except Exception as e:
        print(f"Error loading credentials: {e}")
        return None, None

def save_credentials(site, email, password, credentials_file=DEFAULT_CREDENTIALS_FILE):
    """
    Save credentials for a specific site to the credentials file.
    
    Args:
        site (str): The site to save credentials for
        email (str): Email/username
        password (str): Password
        credentials_file (str): Path to the credentials file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load existing credentials or create new dict
        if os.path.exists(credentials_file):
            with open(credentials_file, 'r') as f:
                credentials = json.load(f)
        else:
            credentials = {}
        
        # Update credentials for the specified site
        if site not in credentials:
            credentials[site] = {}
            
        credentials[site]['email'] = email
        credentials[site]['password'] = password
        
        # Save updated credentials
        with open(credentials_file, 'w') as f:
            json.dump(credentials, f, indent=4)
            
        print(f"Credentials for '{site}' saved to '{credentials_file}'")
        return True
        
    except Exception as e:
        print(f"Error saving credentials: {e}")
        return False

if __name__ == "__main__":
    # Simple CLI for managing credentials
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage credentials for web scrapers")
    parser.add_argument("--site", default="amazon", help="Site to manage credentials for")
    parser.add_argument("--email", help="Email/username to save")
    parser.add_argument("--password", help="Password to save")
    parser.add_argument("--file", default=DEFAULT_CREDENTIALS_FILE, help="Credentials file path")
    
    args = parser.parse_args()
    
    if args.email and args.password:
        # Save mode
        save_credentials(args.site, args.email, args.password, args.file)
    else:
        # Load mode
        email, password = load_credentials(args.site, args.file)
        if email and password:
            print(f"Found credentials for '{args.site}':")
            print(f"Email: {email}")
            print(f"Password: {'*' * len(password)}")
        else:
            print(f"No valid credentials found for '{args.site}'")
