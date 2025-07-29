#!/usr/bin/env python3
"""
Interactive Amazon Review Scraper

This script provides an interactive browser session to help with scraping reviews
from Amazon and other e-commerce sites. It opens a visible browser window and
allows the user to solve CAPTCHAs if needed.
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

# Import credentials manager
from credentials_manager import load_credentials

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class InteractiveReviewScraper:
    """Interactive scraper for product reviews."""
    
    def __init__(self, headless=False, wait_time=30, email=None, password=None, credentials_file="credentials.json"):
        """Initialize the scraper with interactive options."""
        self.options = Options()
        self.credentials_file = credentials_file
        
        # Try to load credentials from file if not provided
        if not email or not password:
            loaded_email, loaded_password = load_credentials(site="amazon", credentials_file=credentials_file)
            self.email = email or loaded_email
            self.password = password or loaded_password
        else:
            self.email = email
            self.password = password
            
        self.wait_time = wait_time
        
        # For interactive mode, we don't use headless
        if headless:
            print("Note: Headless mode is disabled for interactive scraping")
        
        # Set up Chrome options
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option("useAutomationExtension", False)
        
        # Set a reasonable user agent
        self.options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36")
        
        self.driver = None
        self.wait_time = wait_time  # Time to wait for user interaction
    
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
            
    def handle_login(self):
        """Handle Amazon login if required."""
        try:
            # Check for various login indicators
            login_detected = False
            try:
                # Check for email fields with different possible IDs
                email_fields = self.driver.find_elements(By.CSS_SELECTOR, "input[type='email'], #ap_email, #ap_email_login")
                if email_fields:
                    login_detected = True
            except:
                pass
                
            # Check for login text as fallback
            if not login_detected:
                page_source_lower = self.driver.page_source.lower()
                login_detected = "sign-in" in page_source_lower or "sign in" in page_source_lower or "signin" in self.driver.current_url.lower()
            
            if login_detected:
                print("\n" + "="*50)
                print("LOGIN FORM DETECTED")
                print("="*50)
                
                # Take screenshot for debugging
                screenshot_path = "login_screen.png"
                self.driver.save_screenshot(screenshot_path)
                print(f"Login screen screenshot saved to {screenshot_path}")
                
                # Save HTML for debugging
                with open("login_page.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                print("Saved login page HTML for debugging")
                
                # Load credentials if not already provided
                if not self.email or not self.password:
                    print("No credentials provided. Checking credentials file...")
                    if hasattr(self, 'credentials_file') and self.credentials_file:
                        from credentials_manager import load_credentials
                        self.email, self.password = load_credentials("amazon", self.credentials_file)
                        print(f"Loaded credentials from {self.credentials_file}")
                
                # Still need to prompt if credentials not found
                if not self.email:
                    self.email = input("Enter your Amazon email: ")
                if not self.password:
                    import getpass
                    self.password = getpass.getpass("Enter your Amazon password: ")
                
                # STEP 1: Handle email input
                print("\nAttempting to enter email...")
                email_entered = False
                
                # Try different methods to find and fill email field
                for selector in ["#ap_email_login", "#ap_email", "input[type='email']"]:
                    try:
                        email_field = self.driver.find_element(By.CSS_SELECTOR, selector)
                        print(f"Found email field using selector: {selector}")
                        email_field.clear()
                        email_field.send_keys(self.email)
                        email_entered = True
                        break
                    except Exception:
                        continue
                
                if not email_entered:
                    print("Could not find email field. Please check the browser and enter email manually.")
                    input("Press Enter after entering email...")
                else:
                    print(f"Email entered: {self.email}")
                
                # Try to find and click continue button
                print("\nAttempting to click continue button...")
                button_clicked = False
                
                # Try different methods to find continue button - based on user's observation
                try:
                    # First try the specific structure described by the user
                    continue_span = self.driver.find_element(By.ID, "continue")
                    print("Found continue span element")
                    
                    # Try to find the input element inside or near this span
                    try:
                        # Look for the input inside the span
                        continue_button = continue_span.find_element(By.CSS_SELECTOR, "input.a-button-input[type='submit']")
                    except:
                        # If not found inside, try to find it nearby with the same attributes
                        continue_button = self.driver.find_element(By.CSS_SELECTOR, "input.a-button-input[type='submit']")
                    
                    print("Found continue button inside/near span")
                    continue_button.click()
                    button_clicked = True
                except Exception as e:
                    print(f"Could not find button using span approach: {e}")
                    
                    # Fall back to trying various selectors if the specific approach failed
                    for selector in ["#continue input", "span#continue input", "input.a-button-input[type='submit']", "#continue", "input[type='submit']", "button[type='submit']", ".a-button-input"]:
                        try:
                            continue_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                            print(f"Found continue button using selector: {selector}")
                            continue_button.click()
                            button_clicked = True
                            break
                        except Exception:
                            continue
                
                if not button_clicked:
                    print("Could not find continue button. Please click it manually.")
                    input("Press Enter after clicking continue...")
                
                # Wait for transition to password page
                print("\nWaiting for password field to appear...")
                time.sleep(3)  # Give time for page transition
                
                # STEP 2: Handle password input
                print("\nAttempting to enter password...")
                password_entered = False
                
                # Try different methods to find and fill password field
                for selector in ["#ap_password", "input[type='password']"]:
                    try:
                        password_field = self.driver.find_element(By.CSS_SELECTOR, selector)
                        print(f"Found password field using selector: {selector}")
                        password_field.clear()
                        password_field.send_keys(self.password)
                        password_entered = True
                        break
                    except Exception:
                        continue
                
                if not password_entered:
                    print("Could not find password field. Please check the browser and enter password manually.")
                    input("Press Enter after entering password...")
                else:
                    print("Password entered")
                
                # Try to find and click sign-in button
                print("\nAttempting to click sign-in button...")
                signin_clicked = False
                
                # Try different methods to find sign-in button
                for selector in ["#signInSubmit", "input[type='submit']", "button[type='submit']", ".a-button-input"]:
                    try:
                        signin_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                        print(f"Found sign-in button using selector: {selector}")
                        signin_button.click()
                        signin_clicked = True
                        break
                    except Exception:
                        continue
                
                if not signin_clicked:
                    print("Could not find sign-in button. Please click it manually.")
                    input("Press Enter after clicking sign-in...")
                    
                # Take screenshot after login attempt
                self.driver.save_screenshot("after_login.png")
                print("Screenshot saved after login attempt")
                
                # Check for CAPTCHA or verification
                time.sleep(3)  # Wait for page to load
                
                # Wait for potential CAPTCHA or verification
                print("\n" + "="*50)
                print("LOGIN SUBMITTED")
                print("="*50)
                print("Checking for CAPTCHA or additional verification...")
                
                # Check for CAPTCHA indicators
                time.sleep(3)  # Wait for page to load
                if "captcha" in self.driver.page_source.lower() or "puzzle" in self.driver.page_source.lower():
                    print("\nCAPTCHA detected! Please solve it in the browser window.")
                    self.driver.save_screenshot("captcha_screen.png")
                    print(f"CAPTCHA screenshot saved to captcha_screen.png")
                
                print("\nPlease check the browser window and:")
                print("1. Solve any CAPTCHA if present")
                print("2. Complete any additional verification steps")
                print("3. Make sure you can see the Amazon page properly")
                input("Press Enter when you have completed these steps and are ready to continue...")
                
                # Check if login was successful
                if "sign in" in self.driver.page_source.lower() or "sign-in" in self.driver.current_url.lower():
                    print("\nWarning: Still seeing login indicators. Login might have failed.")
                    print("Please check your credentials and try again.")
                else:
                    print("\nLogin process completed. Continuing with scraping...")
                
                return True
            return False
        except Exception as e:
            print(f"\nError during login attempt: {e}")
            print("Please check the browser window and handle login manually if needed.")
            input("Press Enter when you're ready to continue...")
            return True  # Return True to continue with scraping
    
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
    
    def get_reviews(self, url, max_pages=5):
        """Get reviews from an Amazon product page with interactive mode for CAPTCHA solving."""
        reviews = []
        current_page = 1
        
        # Make sure the browser is started
        if not self.driver:
            self.start_browser()
        
        try:
            # Convert product URL to reviews URL
            domain = self.get_domain(url)
            try:
                product_id = self.get_product_id(url)
                reviews_url = f"https://{domain}/product-reviews/{product_id}"
            except ValueError:
                reviews_url = url
            
            while current_page <= max_pages:
                # Construct page URL
                page_url = reviews_url if current_page == 1 else f"{reviews_url}?pageNumber={current_page}"
                print(f"Scraping page {current_page}: {page_url}")
                
                # Load the page
                self.driver.get(page_url)
                
                # Check for login form first
                login_detected = self.handle_login()
                if login_detected:
                    print("Login handled. Continuing with review scraping...")
                
                # Wait for reviews to load with a longer timeout
                try:
                    WebDriverWait(self.driver, 30).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 
                            ".reviews-content, #cm_cr-review_list, li[data-hook='review']"))
                    )
                except Exception as e:
                    print(f"Warning: Timeout waiting for reviews to load: {e}")
                    
                    # Check for CAPTCHA
                    if "captcha" in self.driver.page_source.lower() or "puzzle" in self.driver.page_source.lower():
                        print("\nCAPTCHA detected. Please solve it manually.")
                    
                    # Check for login requirement again (sometimes appears after CAPTCHA)
                    if "sign in" in self.driver.page_source.lower() or "sign-in" in self.driver.page_source.lower():
                        print("\nLogin required. Attempting to log in...")
                        self.handle_login()
                
                print("\nCAPTCHA or verification may be required.")
                print("Please solve any CAPTCHA or verification in the browser window.")
                print("Press Enter when you have solved the CAPTCHA and the reviews are visible...")
                input()
                
                # Wait a bit more after user input
                time.sleep(5)
                
                print(f"\nWaiting {self.wait_time} seconds for page to load and for you to solve any CAPTCHAs...")
                print("If you see a CAPTCHA, please solve it now.")
                print("Press Ctrl+C in the terminal to stop the scraping process at any time.")
                time.sleep(self.wait_time)  # Wait for page to load and for user to solve CAPTCHA if needed
                
                # Save cookies after successful page load for future use
                if current_page == 1:
                    self.save_cookies()
                
                # Parse the page with BeautifulSoup
                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                
                # Find the main reviews container
                reviews_content = soup.select_one(".reviews-content")
                if not reviews_content:
                    print("Could not find the reviews-content container.")
                    
                    # In interactive mode, ask the user what to do
                    print("\nCould not find the reviews-content container. What would you like to do?")
                    print("1. Save the HTML and screenshot for debugging and continue")
                    print("2. Try to continue anyway")
                    print("3. Skip this page and try the next one")
                    print("4. Abort scraping")
                    choice = input("Enter your choice (1-4): ")
                    
                    if choice == "1":
                        # Save HTML for debugging
                        with open(f"page_{current_page}.html", "w", encoding="utf-8") as f:
                            f.write(self.driver.page_source)
                        print(f"Saved HTML to page_{current_page}.html for debugging")
                        
                        # Save screenshot for debugging
                        try:
                            self.driver.save_screenshot(f"page_{current_page}_screenshot.png")
                            print(f"Saved screenshot to page_{current_page}_screenshot.png")
                        except Exception as ss_error:
                            print(f"Could not save screenshot: {ss_error}")
                    elif choice == "3":
                        print("Skipping this page...")
                        current_page += 1
                        continue
                    elif choice == "4":
                        print("Aborting scraping...")
                        break
                
                # Find the review list container
                review_list_container = soup.select_one("#cm_cr-review_list")
                if not review_list_container:
                    print("Could not find the review list container.")
                    
                    # In interactive mode, ask the user what to do
                    print("\nCould not find the review list container. What would you like to do?")
                    print("1. Save the HTML and screenshot for debugging and continue")
                    print("2. Try to continue anyway")
                    print("3. Skip this page and try the next one")
                    print("4. Abort scraping")
                    choice = input("Enter your choice (1-4): ")
                    
                    if choice == "1":
                        # Save HTML for debugging
                        with open(f"page_{current_page}.html", "w", encoding="utf-8") as f:
                            f.write(self.driver.page_source)
                        print(f"Saved HTML to page_{current_page}.html for debugging")
                        
                        # Save screenshot for debugging
                        try:
                            self.driver.save_screenshot(f"page_{current_page}_screenshot.png")
                            print(f"Saved screenshot to page_{current_page}_screenshot.png")
                        except Exception as ss_error:
                            print(f"Could not save screenshot: {ss_error}")
                    elif choice == "3":
                        print("Skipping this page...")
                        current_page += 1
                        continue
                    elif choice == "4":
                        print("Aborting scraping...")
                        break
                
                # Get all review items
                review_items = soup.select("li[data-hook='review'].review")
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
                        
                        # Get review date
                        date = ""
                        date_elem = review_item.select_one("span[data-hook='review-date']")
                        if date_elem:
                            date = date_elem.text.strip()
                        
                        # Get verified purchase status
                        verified = bool(review_item.select_one("span[data-hook='avp-badge']") 
                                        or review_item.select_one(".verified-purchase"))
                        
                        # Get review body
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
                        
                        # Get helpful votes
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
                
                # Check for next page
                next_button = soup.select_one("li.a-last a, .a-pagination .a-last a")
                if not next_button:
                    print("No next page button found.")
                    break
                
                # Ask user if they want to continue to the next page
                if current_page < max_pages:
                    user_input = input(f"Continue to page {current_page + 1}? (y/n): ")
                    if user_input.lower() != 'y':
                        break
                
                current_page += 1
                
        except KeyboardInterrupt:
            print("\nScraping interrupted by user.")
        except Exception as e:
            print(f"Error scraping reviews: {e}")
            import traceback
            traceback.print_exc()
        
        return reviews
    
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


def main():
    """Main function to run the scraper."""
    parser = argparse.ArgumentParser(description="Interactive scraper for product reviews")
    parser.add_argument("url", help="URL of the product page")
    parser.add_argument("--output", "-o", help="Output file name (without extension)")
    parser.add_argument("--format", "-f", choices=["csv", "json"], default="csv", help="Output format (csv or json)")
    parser.add_argument("--pages", "-p", type=int, default=5, help="Maximum number of pages to scrape")
    parser.add_argument("--wait", "-w", type=int, default=30, help="Wait time in seconds for user interaction")
    parser.add_argument("--email", "-e", help="Amazon account email for login if required")
    parser.add_argument("--password", help="Amazon account password for login if required")
    parser.add_argument("--credentials", "-c", default="credentials.json", help="Path to credentials file (default: credentials.json)")
    
    args = parser.parse_args()
    
    try:
        # Initialize the scraper with login credentials if provided
        scraper = InteractiveReviewScraper(headless=False, wait_time=args.wait, 
                                         email=args.email, password=args.password,
                                         credentials_file=args.credentials)
        
        # Get reviews
        reviews = scraper.get_reviews(args.url, max_pages=args.pages)
        
        # Save reviews
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
