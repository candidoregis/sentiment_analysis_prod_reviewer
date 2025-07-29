#!/usr/bin/env python3
"""
Amazon Product Review Scraper

A simplified version that uses Selenium with additional anti-detection measures
to scrape product reviews from Amazon.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from urllib.parse import urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager

# Import credentials manager from utils
from src.utils.credentials_manager import load_credentials, save_credentials

# Helper function to get debug file paths
def debug_get_file_path(filename):
    """Returns the path to a debug file in the debug directory"""
    debug_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')), 'debug')
    # Create debug directory if it doesn't exist
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    return os.path.join(debug_dir, filename)

class AmazonReviewScraper:
    """Scraper for Amazon product reviews with improved anti-detection."""
    
    def __init__(self, headless=True, email=None, password=None, credentials_file=None):
        """Initialize the Amazon scraper with anti-bot detection measures."""
        self.options = Options()
        
        # Set default credentials file path in the config directory
        if credentials_file is None:
            self.credentials_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')), 'config', 'credentials.json')
        else:
            self.credentials_file = credentials_file
        
        # Try to load credentials from file if not provided
        if not email or not password:
            loaded_email, loaded_password = load_credentials(site="amazon", credentials_file=self.credentials_file)
            self.email = email or loaded_email
            self.password = password or loaded_password
        else:
            self.email = email
            self.password = password
        
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
            self.driver = webdriver.Chrome(options=self.options)
            print("Successfully initialized Chrome browser")
            
            # Execute CDP commands to prevent detection
            self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                """
            })
            
        except Exception as e:
            print(f"Error initializing Chrome: {e}")
            sys.exit(1)
            
        return self.driver
    
    def close_browser(self):
        """Close the browser."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def get_product_id(self, url):
        """Extract product ID from Amazon URL."""
        if "/dp/" in url:
            return url.split("/dp/")[1].split("/")[0].split("?")[0]
        elif "/product/" in url:
            return url.split("/product/")[1].split("/")[0].split("?")[0]
        else:
            raise ValueError("Could not extract product ID from URL")
    
    def get_domain(self, url):
        """Extract domain from Amazon URL."""
        parsed_url = urlparse(url)
        return parsed_url.netloc
        
    def random_delay(self, min_seconds=2, max_seconds=7):
        """Add a random delay between requests to avoid detection."""
        delay = random.uniform(min_seconds, max_seconds)
        print(f"Waiting {delay:.2f} seconds between requests...")
        time.sleep(delay)
    
    def debug_load_cookies(self, cookie_file=None):
        """Load cookies from a file if it exists."""
        if not self.driver:
            return False
        
        # Use config directory for cookie file if not specified
        if cookie_file is None:
            cookie_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')), 'config', 'amazon_cookies.json')
            
        if os.path.exists(cookie_file):
            try:
                with open(cookie_file, "r") as f:
                    cookies = json.load(f)
                    for cookie in cookies:
                        try:
                            self.driver.add_cookie(cookie)
                        except Exception as e:
                            print(f"Error adding cookie: {e}")
                print(f"Loaded cookies from {cookie_file}")
                return True
            except Exception as e:
                print(f"Error loading cookies: {e}")
        return False
    
    def debug_save_cookies(self, cookie_file=None):
        """Save cookies to a file for later use."""
        if self.driver:
            # Use config directory for cookie file if not specified
            if cookie_file is None:
                cookie_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')), 'config', 'amazon_cookies.json')
            try:
                cookies = self.driver.get_cookies()
                with open(cookie_file, "w") as f:
                    json.dump(cookies, f)
                print(f"Saved cookies to {cookie_file}")
                return True
            except Exception as e:
                print(f"Error saving cookies: {e}")
        return False
        
    def simulate_human_behavior(self):
        """Simulate human-like behavior to avoid bot detection."""
        if not self.driver:
            return
            
        try:
            # Scroll down slowly
            for i in range(3):
                scroll_amount = random.randint(300, 700)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(0.5, 1.5))
                
            # Scroll back up a bit
            self.driver.execute_script(f"window.scrollBy(0, -{random.randint(100, 300)});")
            time.sleep(random.uniform(0.5, 1.0))
            
        except Exception as e:
            print(f"Error during human behavior simulation: {e}")
    
    def handle_login(self):
        """Handle Amazon login if required."""
        try:
            # Check for login form with various possible indicators
            login_detected = False
            try:
                login_elements_1 = self.driver.find_elements(By.ID, "ap_email_login")
                login_elements_2 = self.driver.find_elements(By.ID, "ap_email")
                login_detected = len(login_elements_1) > 0 or len(login_elements_2) > 0
            except:
                # Check for login in page source as fallback
                login_detected = "sign-in" in self.driver.page_source.lower() or "sign in" in self.driver.page_source.lower()
            
            if login_detected:
                print("\nLogin form detected. Attempting to log in...")
                
                # Take screenshot for debugging
                try:
                    screenshot_path = debug_get_file_path("login_screen.png")
                    self.driver.save_screenshot(screenshot_path)
                    print(f"Login screen screenshot saved to {screenshot_path}")
                except Exception as e:
                    print(f"Could not save screenshot: {e}")
                
                # Save initial login page HTML for debugging
                try:
                    with open(debug_get_file_path("login_page_initial.html"), "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    print("Saved initial login page HTML for debugging")
                except Exception as e:
                    print(f"Could not save HTML: {e}")
                
                # If email/password not provided, can't proceed in headless mode
                if not self.email or not self.password:
                    print("Error: Login required but no credentials provided. Cannot proceed in headless mode.")
                    print("Please update credentials.json or use interactive_review_scraper.py instead.")
                    return False
                
                # Step 1: Enter email - try both possible field IDs
                try:
                    # Wait for email field to be present and visible
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "#ap_email_login, #ap_email"))
                    )
                    
                    try:
                        email_field = self.driver.find_element(By.ID, "ap_email_login")
                    except:
                        email_field = self.driver.find_element(By.ID, "ap_email")
                        
                    print("Found email field, entering email...")
                    email_field.clear()
                    email_field.send_keys(self.email)
                    time.sleep(1)  # Small pause after typing
                    
                    # Find and click continue button - try multiple approaches
                    continue_button = None
                    selectors = [
                        "#continue input", 
                        "span#continue input", 
                        "input.a-button-input[type='submit']", 
                        "#continue", 
                        "input[type='submit']", 
                        "button[type='submit']", 
                        ".a-button-input"
                    ]
                    
                    # Try each selector until we find the continue button
                    for selector in selectors:
                        try:
                            continue_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                            )
                            print(f"Found continue button using selector: {selector}")
                            break
                        except Exception:
                            continue
                    
                    if not continue_button:
                        print("Could not find continue button with any selector")
                        raise Exception("Continue button not found")
                    
                    print("Clicking continue button...")
                    # Use JavaScript to click the button to avoid potential interception issues
                    self.driver.execute_script("arguments[0].click();", continue_button)
                    
                    # Save HTML after clicking continue
                    try:
                        self.driver.save_screenshot(debug_get_file_path("after_continue_click.png"))
                        with open(debug_get_file_path("after_continue_click.html"), "w", encoding="utf-8") as f:
                            f.write(self.driver.page_source)
                        print("Saved HTML after clicking continue button")
                    except Exception as e:
                        print(f"Could not save after-continue state: {e}")
                    
                except Exception as e:
                    print(f"Error in email step: {e}")
                    return False
                
                # Wait for page to stabilize after clicking continue
                time.sleep(3)
                
                # Check if we're on a CAPTCHA page
                if "captcha" in self.driver.page_source.lower() or "puzzle" in self.driver.page_source.lower():
                    print("CAPTCHA detected after email step! Cannot proceed in headless mode.")
                    print("Please use interactive_review_scraper.py instead.")
                    try:
                        self.driver.save_screenshot(debug_get_file_path("captcha_detected.png"))
                        print("Screenshot saved to captcha_detected.png")
                    except Exception:
                        pass
                    return False
                
                # Wait for password field with increased timeout and better error handling
                try:
                    print("Waiting for password field...")
                    # First check if we're still on the same page or if we got redirected
                    current_url = self.driver.current_url
                    print(f"Current URL after continue: {current_url}")
                    
                    # Try multiple approaches to find the password field
                    password_field = None
                    password_selectors = ["#ap_password", "input[type='password']", "input[name='password']", ".a-input-text"]
                    
                    for selector in password_selectors:
                        try:
                            print(f"Looking for password field with selector: {selector}")
                            password_field = WebDriverWait(self.driver, 5).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                            )
                            print(f"Found password field with selector: {selector}")
                            break
                        except Exception:
                            continue
                    
                    if not password_field:
                        print("Could not find password field with any selector")
                        raise Exception("Password field not found")
                    
                    # Give a moment for the page to stabilize
                    time.sleep(2)
                    
                    # Enter password
                    print("Entering password...")
                    # Use JavaScript to set the value and then trigger events
                    self.driver.execute_script(
                        "arguments[0].value = arguments[1]; "
                        "arguments[0].dispatchEvent(new Event('change')); "
                        "arguments[0].dispatchEvent(new Event('input'));", 
                        password_field, self.password
                    )
                    time.sleep(1)  # Small pause after typing
                    
                    # Find and click sign-in button
                    signin_button = None
                    signin_selectors = ["#signInSubmit", "input[type='submit']", "button[type='submit']", ".a-button-input"]
                    
                    for selector in signin_selectors:
                        try:
                            signin_button = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                            )
                            print(f"Found sign-in button with selector: {selector}")
                            break
                        except Exception:
                            continue
                    
                    if not signin_button:
                        print("Could not find sign-in button with any selector")
                        raise Exception("Sign-in button not found")
                    
                    print("Clicking sign-in button...")
                    # Use JavaScript to click the button
                    self.driver.execute_script("arguments[0].click();", signin_button)
                    
                except Exception as e:
                    print(f"Error in password step: {e}")
                    # Try direct navigation to the reviews page as a fallback
                    print("Login failed. Attempting to continue anyway...")
                    return False
                
                print("Login submitted. Waiting for page to load...")
                time.sleep(5)  # Wait for login to complete
                
                # Check for CAPTCHA
                if "captcha" in self.driver.page_source.lower() or "puzzle" in self.driver.page_source.lower():
                    print("CAPTCHA detected! Cannot proceed in headless mode.")
                    print("Please use interactive_review_scraper.py instead.")
                    try:
                        self.driver.save_screenshot(debug_get_file_path("captcha_detected.png"))
                        print("Screenshot saved to captcha_detected.png")
                    except Exception:
                        pass
                    return False
                
                # Save HTML for debugging
                try:
                    with open(debug_get_file_path("login_page_final.html"), "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    print("Saved final login page HTML for debugging")
                except Exception as e:
                    print(f"Could not save final login state: {e}")
                
                return True
            
            return False
        except Exception as e:
            print(f"Error during login: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_product_title(self):
        """Get the product title from the page."""
        try:
            # First try the product info section on reviews page
            try:
                # Look for the product info section with the specific structure mentioned
                product_info = self.driver.find_element(By.ID, "cm_cr-product_info")
                product_link = product_info.find_element(By.CSS_SELECTOR, "h1 a[data-hook='product-link']")
                return product_link.text.strip()
            except Exception as e:
                print(f"Could not find title in cm_cr-product_info: {e}")
                
                # Fall back to other methods
                try:
                    title_element = self.driver.find_element(By.ID, "productTitle")
                    return title_element.text.strip()
                except:
                    # Try other possible selectors
                    selectors = [
                        ".product-title", 
                        ".review-product-title",
                        "#title",
                        ".product-title-word-break"
                    ]
                    
                    for selector in selectors:
                        try:
                            title_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                            title = title_element.text.strip()
                            if title:
                                return title
                        except:
                            continue
            
            # If we get here, we couldn't find the title
            print("Could not find product title with any selector")
            return "Unknown Product"
        except Exception as e:
            print(f"Error getting product title: {e}")
            return "Unknown Product"
    
    def get_reviews(self, url, max_pages=5, max_reviews=50):
        """Get reviews from Amazon product page.
        
        Args:
            url (str): URL of the Amazon product or reviews page
            max_pages (int): Maximum number of pages to scrape
            max_reviews (int): Maximum number of reviews to collect
            
        Returns:
            list: List of review dictionaries
        """
        reviews = []
        product_title = ""
        
        # Make sure the browser is started
        if not self.driver:
            print("Browser not started. Starting browser now...")
            self.start_browser()
        
        # Convert product URL to reviews URL if needed
        if "/dp/" in url and "/reviews/" not in url:
            asin = re.search(r'/dp/([A-Z0-9]{10})', url)
            if asin:
                asin = asin.group(1)
                # Extract the domain from the original URL
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                url = f"https://{domain}/product-reviews/{asin}/"
                print(f"Converted to reviews URL: {url}")
        
        try:
            print(f"Navigating to {url}")
            self.driver.get(url)
            
            # Check if login is required
            login_success = self.handle_login()
            if login_success:
                print("Login successful, reloading review page...")
                # Reload the page to ensure we're authenticated
                self.driver.get(url)
            
            # Save cookies for future use
            try:
                self.debug_save_cookies()
            except Exception as e:
                print(f"Could not save cookies: {e}")
            
            # Add random wait between requests to avoid detection
            wait_time = random.uniform(2, 5)
            print(f"Waiting {wait_time:.2f} seconds between requests...")
            time.sleep(wait_time)
            
            # Check if browser is still open
            try:
                # Just a simple check to see if the driver is still responsive
                current_url = self.driver.current_url
            except Exception as e:
                print(f"Browser window appears to be closed: {e}")
                print("Attempting to restart browser session...")
                try:
                    self.start_browser()
                    self.driver.get(url)
                    print("Successfully restarted browser session")
                except Exception as restart_error:
                    print(f"Failed to restart browser: {restart_error}")
                    return reviews
            
            # Wait for reviews to load using multiple possible selectors
            review_container = None
            selectors = [
                ".reviews-content",
                "#cm_cr-review_list",
                "li[data-hook='review']",
                ".review",
                "div[data-hook='review']",
                ".a-section.review"
            ]
            
            for selector in selectors:
                try:
                    print(f"Waiting for reviews using selector: {selector}")
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    review_container = self.driver.find_element(By.CSS_SELECTOR, selector)
                    print(f"Found reviews with selector: {selector}")
                    break
                except Exception:
                    continue
            
            if not review_container:
                # Check if there's a "See all reviews" link
                try:
                    see_all_reviews = self.driver.find_element(By.XPATH, "//a[contains(text(), 'See all reviews')]")
                    print("Found 'See all reviews' link, clicking it...")
                    # Use JavaScript to click the link to avoid potential issues
                    self.driver.execute_script("arguments[0].click();", see_all_reviews)
                    time.sleep(3)  # Wait for page to load
                    
                    # Try again with the selectors
                    for selector in selectors:
                        try:
                            print(f"Waiting for reviews using selector: {selector}")
                            WebDriverWait(self.driver, 10).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                            )
                            review_container = self.driver.find_element(By.CSS_SELECTOR, selector)
                            print(f"Found reviews with selector: {selector}")
                            break
                        except Exception:
                            continue
                except Exception:
                    print("No 'See all reviews' link found")
            
            if not review_container:
                print("Warning: Timeout waiting for reviews to load with any selector")
                # Save the HTML for debugging
                try:
                    debug_html_path = debug_get_file_path("debug_reviews_page.html")
                    with open(debug_html_path, "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    print(f"Saved debug HTML to {debug_html_path}")
                    
                    # Take a screenshot for debugging
                    screenshot_path = debug_get_file_path("debug_reviews_page.png")
                    self.driver.save_screenshot(screenshot_path)
                    print(f"Saved screenshot to {screenshot_path}")
                except Exception as e:
                    print(f"Could not save debug files: {e}")
                
                # Try to find reviews with a more generic approach
                try:
                    # Look for any elements that might contain reviews
                    review_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                        "div.review, div[data-hook*='review'], li[data-hook*='review'], .a-section.review")
                    
                    if review_elements:
                        print(f"Found {len(review_elements)} potential review elements with generic selector")
                    else:
                        print("No reviews found with generic selector either")
                        return reviews
                except Exception as e:
                    print(f"Error finding reviews with generic selector: {e}")
                    return reviews
            # Extract product title once
            try:
                product_title_element = self.driver.find_element(By.CSS_SELECTOR, 
                    "h1.a-size-large, .product-title, h1[data-hook='product-title'], .product-title-word-break")
                product_title = product_title_element.text.strip()
                print(f"Product title: {product_title}")
            except Exception as e:
                print(f"Could not extract product title: {e}")
                # Try alternative selectors
                try:
                    product_title_element = self.driver.find_element(By.CSS_SELECTOR, "h1")
                    product_title = product_title_element.text.strip()
                    print(f"Product title (alternative): {product_title}")
                except Exception:
                    print("Could not extract product title with alternative selector")
                    product_title = "Unknown Product"

            # Process multiple pages of reviews
            page_num = 1
            total_reviews_collected = 0
            
            while page_num <= max_pages and total_reviews_collected < max_reviews:
                print(f"\nProcessing review page {page_num} (collected {total_reviews_collected}/{max_reviews} reviews so far)")
                
                # Check if browser is still responsive
                try:
                    # Simple check
                    _ = self.driver.current_url
                except Exception as e:
                    print(f"Browser window closed during review extraction: {e}")
                    break
                
                # Find all review elements on the current page
                review_elements = []
                selectors = [
                    "div[data-hook='review']",
                    "li[data-hook='review']",
                    ".review",
                    ".a-section.review",
                    "div[id^='customer_review-']"
                ]
                
                for selector in selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            review_elements = elements
                            print(f"Found {len(review_elements)} reviews with selector: {selector}")
                            break
                    except Exception as e:
                        print(f"Error with selector {selector}: {e}")
                        continue
                
                if not review_elements:
                    print("No review elements found on this page")
                    break
                
                # Process each review
                for i, review_element in enumerate(review_elements):
                    
                    try:
                        review_data = {}
                        
                        # Add product title to the review data
                        review_data["product_title"] = product_title
                        
                        # Extract review title
                        try:
                            title_element = review_element.find_element(By.CSS_SELECTOR, 
                                "a.review-title, span.review-title, [data-hook='review-title']")
                            review_data["title"] = title_element.text.strip()
                        except Exception:
                            try:
                                # Alternative selector
                                title_element = review_element.find_element(By.CSS_SELECTOR, 
                                    ".a-size-base.review-title, .a-size-medium.review-title")
                                review_data["title"] = title_element.text.strip()
                            except Exception:
                                review_data["title"] = "No title found"
                        
                        # Extract rating - try multiple approaches
                        try:
                            # First try: Look for data-hook='review-star-rating'
                            rating_element = review_element.find_element(By.CSS_SELECTOR, 
                                "[data-hook='review-star-rating'], [data-hook='cmps-review-star-rating']")
                            rating_text = rating_element.get_attribute("textContent").strip()
                            rating_match = re.search(r'(\d\.\d|\d) out of \d', rating_text)
                            if rating_match:
                                review_data["rating"] = float(rating_match.group(1))
                            else:
                                # Try to extract just the number
                                rating_match = re.search(r'(\d\.\d|\d)', rating_text)
                                if rating_match:
                                    review_data["rating"] = float(rating_match.group(1))
                                else:
                                    review_data["rating"] = 0.0
                        except Exception:
                            try:
                                # Second try: Look for i.a-star-{rating} class
                                star_element = review_element.find_element(By.CSS_SELECTOR, "i[class*='a-star-']")
                                class_name = star_element.get_attribute("class")
                                rating_match = re.search(r'a-star-(\d)', class_name)
                                if rating_match:
                                    review_data["rating"] = float(rating_match.group(1))
                                else:
                                    review_data["rating"] = 0.0
                            except Exception:
                                try:
                                    # Third try: Look for aria-label with rating
                                    star_element = review_element.find_element(By.CSS_SELECTOR, 
                                        "span[class*='a-icon-star'], i[class*='a-icon-star']")
                                    aria_label = star_element.get_attribute("aria-label")
                                    if aria_label:
                                        rating_match = re.search(r'(\d\.\d|\d) out of \d', aria_label)
                                        if rating_match:
                                            review_data["rating"] = float(rating_match.group(1))
                                        else:
                                            review_data["rating"] = 0.0
                                    else:
                                        review_data["rating"] = 0.0
                                except Exception:
                                    review_data["rating"] = 0.0
                        
                        # Extract review body
                        try:
                            body_element = review_element.find_element(By.CSS_SELECTOR, 
                                "span[data-hook='review-body'], div[data-hook='review-collapsed']")
                            review_data["body"] = body_element.text.strip()
                        except Exception:
                            try:
                                # Alternative selector
                                body_element = review_element.find_element(By.CSS_SELECTOR, 
                                    ".review-text, .review-text-content")
                                review_data["body"] = body_element.text.strip()
                            except Exception:
                                try:
                                    # Another alternative
                                    body_element = review_element.find_element(By.CSS_SELECTOR, 
                                        ".a-expander-content")
                                    review_data["body"] = body_element.text.strip()
                                except Exception:
                                    review_data["body"] = "No review body found" 
                        
                        # Add review to list if we haven't reached the limit
                        if total_reviews_collected < max_reviews:
                            reviews.append(review_data)
                            total_reviews_collected += 1
                            print(f"Extracted review {i+1}/{len(review_elements)} ({total_reviews_collected}/{max_reviews} total): {review_data['title'][:30]}...")
                            
                            # If we've reached the limit, break out of the loop
                            if total_reviews_collected >= max_reviews:
                                print(f"Reached maximum number of reviews ({max_reviews}). Stopping extraction.")
                                break
                        
                    except Exception as e:
                        print(f"Error processing review {i+1}: {e}")
                
                # Check if there's a next page and navigate to it
                if page_num < max_pages and total_reviews_collected < max_reviews:
                    try:
                        # First check if the next button is disabled
                        try:
                            disabled_next = self.driver.find_element(By.CSS_SELECTOR, 
                                "li.a-disabled.a-last, a.a-disabled.a-last")
                            print("Next page button is disabled. No more pages available.")
                            break
                        except Exception:
                            # Next button is not disabled, try to find and click it
                            next_button = self.driver.find_element(By.CSS_SELECTOR, 
                                "li.a-last a, a.a-last, a[data-hook='pagination-next']")
                            print("Found next page button, clicking...")
                            # Use JavaScript to click the button
                            self.driver.execute_script("arguments[0].click();", next_button)
                            time.sleep(3)  # Wait for next page to load
                            page_num += 1
                    except Exception as e:
                        print(f"No next page button found or error navigating: {e}")
                        break
                else:
                    if total_reviews_collected >= max_reviews:
                        print(f"Reached maximum number of reviews ({max_reviews}). Stopping pagination.")
                    else:
                        print(f"Reached maximum number of pages ({max_pages}). Stopping pagination.")
                    break
            
            return reviews
            
        except Exception as e:
            print(f"Error scraping reviews: {e}")
            # Save debug info
            try:
                # Check if driver is still available
                if self.driver:
                    debug_html_path = debug_get_file_path("error_page.html")
                    with open(debug_html_path, "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    print(f"Saved error page HTML to {debug_html_path}")
                    
                    screenshot_path = debug_get_file_path("error_page.png")
                    self.driver.save_screenshot(screenshot_path)
                    print(f"Saved error screenshot to {screenshot_path}")
                else:
                    print("Cannot save debug info: browser is not available")
            except Exception as screenshot_error:
                print(f"Could not save debug info: {screenshot_error}")
            
            return reviews
                
        except Exception as e:
            print(f"Error scraping reviews: {e}")
            import traceback
            traceback.print_exc()
        
        return reviews
    
    def scroll_page(self):
        """Scroll the page slowly to simulate human behavior."""
        total_height = self.driver.execute_script("return document.body.scrollHeight")
        for i in range(1, 10):
            self.driver.execute_script(f"window.scrollTo(0, {total_height * i / 10});")
            time.sleep(random.uniform(0.1, 0.3))
    
    def export_reviews(self, reviews, output_format='csv', output_file=None):
        """Save reviews to a file in the specified format."""
        if not reviews:
            print("No reviews to save.")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"reviews_{timestamp}"
        
        if output_format.lower() == 'csv':
            if not output_file.endswith('.csv'):
                output_file += '.csv'
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = reviews[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for review in reviews:
                    writer.writerow(review)
            
        elif output_format.lower() == 'json':
            if not output_file.endswith('.json'):
                output_file += '.json'
            
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(reviews, jsonfile, indent=4)
        
        print(f"Saved {len(reviews)} reviews to {output_file}")


def main():
    """Main function to run the scraper."""
    parser = argparse.ArgumentParser(description="Amazon Review Scraper")
    parser.add_argument("url", help="URL of the Amazon product page")
    parser.add_argument("--headless", action="store_true", dest="headless", 
                        help="Run in headless mode (default)")
    parser.add_argument("--no-headless", action="store_false", dest="headless", 
                        help="Run in non-headless mode (interactive)")
    parser.set_defaults(headless=True)
    parser.add_argument("--output", "-o", default=None, 
                        help="Output file path (default: auto-generated based on date)")
    parser.add_argument("--format", "-f", choices=["csv", "json"], default="csv", 
                        help="Output format (default: csv)")
    parser.add_argument("--email", help="Amazon account email")
    parser.add_argument("--password", help="Amazon account password")
    parser.add_argument("--credentials", default="credentials.json", 
                        help="Path to credentials file (default: credentials.json)")
    parser.add_argument("--pages", type=int, default=5, 
                        help="Maximum number of pages to scrape (default: 5)")
    parser.add_argument("--max-reviews", type=int, default=50, 
                        help="Maximum number of reviews to collect (default: 50)")
    args = parser.parse_args()
    
    try:
        # Initialize scraper
        scraper = AmazonReviewScraper(headless=args.headless,
                                     email=args.email,
                                     password=args.password,
                                     credentials_file=args.credentials)
        
        # Get reviews
        reviews = scraper.get_reviews(args.url, max_pages=args.pages, max_reviews=args.max_reviews)
        
        # Close the browser
        scraper.close_browser()
        
        if reviews:
            scraper.export_reviews(reviews, args.format, args.output)
        else:
            print("No reviews found.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Make sure to close the browser
        if 'scraper' in locals() and hasattr(scraper, 'driver') and scraper.driver:
            scraper.close_browser()


if __name__ == "__main__":
    main()
