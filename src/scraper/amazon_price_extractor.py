#!/usr/bin/env python3
"""
Amazon Price Extractor

A simple module to extract product prices from Amazon product pages
without modifying the existing Amazon review scraper.
"""

import os
import re
import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from urllib.parse import urlparse

# Import credentials manager from utils
from src.utils.credentials_manager import load_credentials

class AmazonPriceExtractor:
    """Extractor for Amazon product prices."""
    
    def __init__(self, headless=True, credentials_file=None):
        """Initialize the Amazon price extractor with anti-bot detection measures."""
        self.options = Options()
        
        # Set default credentials file path in the config directory
        if credentials_file is None:
            self.credentials_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')), 'config', 'credentials.json')
        else:
            self.credentials_file = credentials_file
        
        # Set up Chrome options for anti-detection
        if headless:
            self.options.add_argument("--headless=new")
        
        # Common options to avoid detection
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option("useAutomationExtension", False)
        
        # Random user agent to avoid detection
        user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.69"
        ]
        self.options.add_argument(f"user-agent={random.choice(user_agents)}")
        
        self.driver = None
    
    def start_browser(self):
        """Start the browser with configured options."""
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service
            
            # Use ChromeDriverManager with specific version and cache_valid_range
            service = Service(ChromeDriverManager(cache_valid_range=30).install())
            self.driver = webdriver.Chrome(service=service, options=self.options)
            
            # Set script to override navigator properties to avoid detection
            self.driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            """)
            
            print("Successfully initialized Chrome browser for price extraction")
            return self.driver
        except Exception as e:
            print(f"Error initializing Chrome browser for price extraction: {e}")
            # Try with a direct path as fallback
            try:
                print("Attempting fallback browser initialization...")
                self.driver = webdriver.Chrome(options=self.options)
                return self.driver
            except Exception as e2:
                print(f"Fallback browser initialization also failed: {e2}")
                return None
    
    def close_browser(self):
        """Close the browser."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def random_delay(self, min_seconds=1, max_seconds=3):
        """Add a random delay between requests to avoid detection."""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
    
    def get_product_id(self, url):
        """Extract product ID from Amazon URL."""
        # Try to find ASIN in URL using regex
        asin_match = re.search(r'/dp/([A-Z0-9]{10})/', url)
        if asin_match:
            return asin_match.group(1)
        
        # Try another pattern
        asin_match = re.search(r'/product/([A-Z0-9]{10})/', url)
        if asin_match:
            return asin_match.group(1)
        
        # Try gp/product pattern
        asin_match = re.search(r'/gp/product/([A-Z0-9]{10})/', url)
        if asin_match:
            return asin_match.group(1)
        
        return None
    
    def get_domain(self, url):
        """Extract domain from Amazon URL."""
        parsed_url = urlparse(url)
        return parsed_url.netloc
    
    def ensure_product_url(self, url):
        """Ensure we have a product URL, not a reviews URL."""
        # If it's a reviews URL, convert it to a product URL
        if "/product-reviews/" in url:
            asin = re.search(r'/product-reviews/([A-Z0-9]{10})', url)
            if asin:
                asin = asin.group(1)
                domain = self.get_domain(url)
                return f"https://{domain}/dp/{asin}/"
        
        # If it's already a product URL, return it
        return url
    
    def get_product_price(self, url):
        """Get the product price from the Amazon product page.
        
        Args:
            url (str): URL of the Amazon product page
            
        Returns:
            float or None: The product price as a float, or None if not found
        """
        # Make sure we have a product URL
        product_url = self.ensure_product_url(url)
        
        # Make sure the browser is started
        if not self.driver:
            print("Browser not started. Starting browser now...")
            self.start_browser()
        
        try:
            print(f"Navigating to {product_url} to extract price")
            self.driver.get(product_url)
            
            # Add random wait to avoid detection
            self.random_delay(1, 3)
            
            # Try multiple selectors for the price
            price_selectors = [
                ".a-price .a-price-whole",  # Main selector
                "#priceblock_ourprice",
                "#priceblock_dealprice",
                ".a-price",
                ".a-color-price",
                ".a-text-price"
            ]
            
            for selector in price_selectors:
                try:
                    price_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    price_text = price_element.text.strip()
                    
                    # Clean up the price text (remove currency symbols, commas, etc.)
                    # Extract just the numeric part
                    price_text = re.sub(r'[^0-9.]', '', price_text)
                    
                    # Try to convert to float
                    if price_text:
                        try:
                            price = float(price_text)
                            print(f"Found product price: ${price:.2f}")
                            return price
                        except ValueError:
                            print(f"Could not convert price text to float: {price_text}")
                except Exception as e:
                    continue
            
            print("Could not find product price with any selector")
            return None
        except Exception as e:
            print(f"Error getting product price: {e}")
            return None
        finally:
            # Don't close the browser here, let the caller decide when to close it
            pass

def extract_price(url, headless=True):
    """
    Convenience function to extract price from an Amazon product URL.
    
    Args:
        url (str): URL of the Amazon product page
        headless (bool): Whether to run the browser in headless mode
        
    Returns:
        float or None: The product price as a float, or None if not found
    """
    extractor = None
    try:
        extractor = AmazonPriceExtractor(headless=headless)
        price = extractor.get_product_price(url)
        if price is not None:
            return price
        else:
            print("Could not extract product price: No price found on the page")
            return None
    except Exception as e:
        print(f"Could not extract product price: {e}")
        return None
    finally:
        if extractor:
            try:
                extractor.close_browser()
            except Exception as e:
                print(f"Error closing browser: {e}")
                pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract price from Amazon product page")
    parser.add_argument("url", help="URL of the Amazon product page")
    parser.add_argument("--visible", action="store_true", help="Run with visible browser")
    
    args = parser.parse_args()
    
    price = extract_price(args.url, headless=not args.visible)
    if price is not None:
        print(f"Product price: ${price:.2f}")
    else:
        print("Could not extract product price")
