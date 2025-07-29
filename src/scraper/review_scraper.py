#!/usr/bin/env python3
"""
Amazon Product Review Scraper

This script uses Selenium to scrape product reviews from a given URL.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from urllib.parse import urlparse
import random
import pandas as pd
from abc import ABC, abstractmethod
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import platform


class ReviewScraper(ABC):
    """Base class for review scrapers."""
    
    # List of common user agents to rotate through
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
    ]
    
    def __init__(self, headless=True):
        """Initialize the scraper."""
        self.headless = headless
        self.driver = None
        self.user_agent = random.choice(self.USER_AGENTS)
        self.options = Options()
        if headless:
            self.options.add_argument("--headless=new")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument(f"user-agent={self.user_agent}")
        self.options.add_argument("--disable-dev-shm-usage")
        
        # Set appropriate user agent
        if platform.system() == "Darwin":  # macOS
            if platform.machine() == "arm64":  # Apple Silicon
                self.options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Apple Silicon Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36")
            else:  # Intel Mac
                self.options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36")
        else:  # Other platforms
            self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36")
        
        self.driver = None
    
    def start_browser(self):
        """Start the browser."""
        try:
            # Add additional options to help bypass detection
            self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
            self.options.add_experimental_option("useAutomationExtension", False)
            self.options.add_argument("--disable-blink-features=AutomationControlled")
            
            self.driver = webdriver.Chrome(options=self.options)
            
            # Execute CDP commands to disable webdriver flags
            # This helps bypass bot detection
            self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                """
            })
            
            print("Successfully initialized Chrome browser")
        except Exception as e:
            print(f"Error initializing Chrome browser: {e}")
            self.driver = None
        # Try Firefox as a fallback
        if not self.driver:
            try:
                print("Trying Firefox as a fallback...")
                from selenium.webdriver.firefox.options import Options as FirefoxOptions
                firefox_options = FirefoxOptions()
                if "--headless=new" in [arg.split('=')[0] + '=new' for arg in self.options.arguments if arg.startswith('--headless')]:
                    firefox_options.add_argument("--headless")
                self.driver = webdriver.Firefox(options=firefox_options)
                print("Successfully initialized Firefox browser")
            except Exception as firefox_error:
                print(f"Error initializing Firefox: {firefox_error}")
                print("Could not initialize any browser. Please make sure Chrome or Firefox is installed.")
                sys.exit(1)
        return self.driver
    
    def close_browser(self):
        """Close the browser."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def random_delay(self, min_seconds=2, max_seconds=7):
        """Add a random delay between requests to avoid detection."""
        delay = random.uniform(min_seconds, max_seconds)
        print(f"Waiting {delay:.2f} seconds between requests...")
        time.sleep(delay)
    
    def load_cookies(self, cookie_file="amazon_cookies.json"):
        """Load cookies from a file if it exists."""
        import json
        if not self.driver:
            return False
            
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
    
    def save_cookies(self, cookie_file="amazon_cookies.json"):
        """Save cookies to a file for later use."""
        import json
        if self.driver:
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
            # Scroll down slowly in a natural way
            height = self.driver.execute_script("return document.body.scrollHeight")
            for i in range(1, 10):
                # Scroll down in steps
                self.driver.execute_script(f"window.scrollTo(0, {height * i / 10});")
                # Random delay between scrolls
                time.sleep(random.uniform(0.1, 0.5))
                
            # Random mouse movements (using JavaScript since we can't use ActionChains in headless mode)
            self.driver.execute_script("""
                var event = new MouseEvent('mousemove', {
                    'view': window,
                    'bubbles': true,
                    'cancelable': true,
                    'clientX': Math.floor(Math.random() * window.innerWidth),
                    'clientY': Math.floor(Math.random() * window.innerHeight)
                });
                document.dispatchEvent(event);
            """)
            
            print("Simulated human-like behavior")
        except Exception as e:
            print(f"Error simulating human behavior: {e}")
    
    def get_reviews(self, url):
        """Get reviews from the given URL."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_reviews(self, reviews, output_format='csv', output_file=None):
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
        return output_file


class AmazonReviewScraper(ReviewScraper):
    """Scraper for Amazon product reviews."""
    
    def __init__(self, headless=True):
        """Initialize the Amazon scraper."""
        super().__init__(headless)
        self.domain = "amazon.com"
    
    def can_handle(self, url):
        """Check if this scraper can handle the given URL."""
        parsed_url = urlparse(url)
        return "amazon" in parsed_url.netloc
    
    def get_reviews(self, url, max_pages=5):
        """Get reviews from the given Amazon product URL using the specific HTML structure."""
        if not self.driver:
            self.start_browser()
        
        # Handle different Amazon domains (amazon.com, amazon.ca, etc.)
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Keep the main product URL as is
        # We'll extract reviews directly from the product page
        
        reviews = []
        current_page = 1
        
        try:
            while current_page <= max_pages:
                page_url = url if current_page == 1 else f"{url}?pageNumber={current_page}"
                print(f"Scraping page {current_page}: {page_url}")
                
                self.driver.get(page_url)
                
                # Wait for reviews to load using the specific structure
                try:
                    WebDriverWait(self.driver, self.wait_time).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".reviews-content, #cm_cr-review_list, li[data-hook='review']"))
                    )
                except Exception as e:
                    print(f"Warning: Timeout waiting for reviews to load: {e}")
                    print("This could be due to CAPTCHA or other verification.")
                
                # Add a small delay to ensure page is fully loaded
                time.sleep(3)
                
                # Parse the page with BeautifulSoup
                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                
                # Find the main reviews container
                reviews_content = soup.select_one(".reviews-content")
                if not reviews_content:
                    print("Could not find the reviews-content container.")
                    
                    # Save the HTML for debugging
                    with open(f"page_{current_page}.html", "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    print(f"Saved HTML to page_{current_page}.html for debugging")
                    
                    # Save screenshot for debugging
                    try:
                        self.driver.save_screenshot(f"page_{current_page}_screenshot.png")
                        print(f"Saved screenshot to page_{current_page}_screenshot.png")
                    except Exception as ss_error:
                        print(f"Could not save screenshot or HTML: {ss_error}")
                    break
                
                # Find the review list container
                review_list_container = soup.select_one("#cm_cr-review_list")
                if not review_list_container:
                    print("Could not find the review list container.")
                    
                    # Save the HTML for debugging
                    with open(f"page_{current_page}.html", "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    print(f"Saved HTML to page_{current_page}.html for debugging")
                    
                    # Save screenshot for debugging
                    try:
                        self.driver.save_screenshot(f"page_{current_page}_screenshot.png")
                        print(f"Saved screenshot to page_{current_page}_screenshot.png")
                    except Exception as ss_error:
                        print(f"Could not save screenshot or HTML: {ss_error}")
                    break
                
                # Get all review items
                review_items = soup.select("li[data-hook='review'].review")
                
                if not review_items:
                    print("No reviews found on this page.")
                    break
                
                print(f"Found {len(review_items)} reviews on page {current_page}")
                
                # Extract product title from the page
                product_title = "Unknown Product"
                product_title_elem = soup.select_one("#productTitle")
                if product_title_elem:
                    product_title = product_title_elem.text.strip()
                
                # Process each review
                for review_item in review_items:
                    try:
                        # Get review ID
                        review_id = review_item.get("id", "")
                        
                        # Get reviewer name
                        reviewer = "Anonymous"
                        reviewer_elem = review_item.select_one("span.a-profile-name")
                        if reviewer_elem:
                            reviewer = reviewer_elem.text.strip()
                        
                        # Get review title and rating from the first part (h5 element)
                        title = ""
                        rating = None
                        
                        # Find the h5 element containing title and rating
                        title_section = review_item.select_one(".a-row h5")
                        if title_section:
                            # Get the title
                            title_elem = title_section.select_one("span[data-hook='review-title'] span")
                            if title_elem:
                                title = title_elem.text.strip()
                            
                            # Get the rating
                            rating_elem = title_section.select_one("i[data-hook='cmps-review-star-rating']")
                            if rating_elem:
                                try:
                                    # Check for star rating class (a-star-1 through a-star-5)
                                    for i in range(1, 6):
                                        if rating_elem.has_attr('class') and f'a-star-{i}' in rating_elem['class']:
                                            rating = float(i)
                                            break
                                    
                                    # If rating not found by class, try to extract from text
                                    if not rating and rating_elem.select_one("span"):
                                        rating_text = rating_elem.select_one("span").text.strip()
                                        if "out of" in rating_text:
                                            rating = float(rating_text.split(" out of")[0])
                                        elif "stars" in rating_text or "star" in rating_text:
                                            try:
                                                rating = float(rating_text.split(" star")[0])
                                            except (ValueError, IndexError):
                                                pass
                                        else:
                                            try:
                                                rating = float(rating_text.split(" ")[0].replace(",", "."))
                                            except (ValueError, IndexError):
                                                pass
                                except (ValueError, IndexError):
                                    pass
 
                        # Get review body from the second part
                        body = ""
                        
                        # Try multiple selectors for the review body
                        # First try the specific structure
                        review_data_section = review_item.select_one(".review-data")
                        if review_data_section:
                            body_section = review_data_section.select_one("span[data-hook='review-body']")
                            if body_section:
                                # Try to get the text from the span inside
                                span_text = body_section.select_one("span")
                                if span_text:
                                    body = span_text.text.strip()
                                else:
                                    # If no span, get text directly
                                    body = body_section.text.strip()
                        
                        # If still no body, try alternative selectors
                        if not body:
                            # Try direct selector
                            body_elem = review_item.select_one("span[data-hook='review-body']")
                            if body_elem:
                                body = body_elem.text.strip()
                        
                        # If still no body, try more generic selectors
                        if not body:
                            body_elem = review_item.select_one(".a-row.a-spacing-small.review-data")
                            if body_elem:
                                body = body_elem.text.strip()
                                
                        # If still no body, try the most generic selector
                        if not body:
                            body_elem = review_item.select_one("span[data-hook='review-body'] span, .review-text")
                            if body_elem:
                                body = body_elem.text.strip()
                        
                        # Create review data dictionary with only the needed fields
                        review_data = {
                            "product_title": product_title,
                            "rating": rating,
                            "title": title,
                            "body": body,
                        }
                        
                        reviews.append(review_data)
                        
                    except Exception as e:
                        print(f"Error extracting review data: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Check if there's a next page
                next_button = soup.select_one("li.a-last a, .a-pagination .a-last a")
                if not next_button:
                    print("No next page button found.")
                    break
                
                current_page += 1
                
        except Exception as e:
            print(f"Error scraping reviews: {e}")
            import traceback
            traceback.print_exc()
        
        return reviews


def get_scraper_for_url(url, headless=True):
    """Get the appropriate scraper for the given URL."""
    # Currently only supporting Amazon, but can be extended
    scrapers = [
        AmazonReviewScraper(headless)
    ]
    
    for scraper in scrapers:
        if scraper.can_handle(url):
            return scraper
    
    raise ValueError(f"No scraper available for URL: {url}")


def main():
    """Main function to run the scraper."""
    parser = argparse.ArgumentParser(description="Scrape product reviews from a given URL")
    parser.add_argument("url", help="URL of the product page")
    parser.add_argument("--output", "-o", help="Output file name (without extension)")
    parser.add_argument("--format", "-f", choices=["csv", "json"], default="csv", help="Output format (csv or json, default: csv)")
    parser.add_argument("--pages", "-p", type=int, default=5, help="Maximum number of pages to scrape")
    parser.add_argument("--no-headless", action="store_true", help="Run in non-headless mode (show browser)")
    
    args = parser.parse_args()
    
    try:
        # Get the appropriate scraper for the URL
        scraper = get_scraper_for_url(args.url, not args.no_headless)
        
        # Get reviews
        reviews = scraper.get_reviews(args.url, max_pages=args.pages)
        
        # Save reviews
        if reviews:
            scraper.export_reviews(reviews, args.format, args.output)
        else:
            print("No reviews found.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Make sure to close the browser
        if 'scraper' in locals() and scraper.driver:
            scraper.close_browser()


if __name__ == "__main__":
    main()
