from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import re

class AlternativeLinkScraper:
    def __init__(self, headless=True):
        self.options = Options()
        self.options.add_argument('--headless') if headless else None
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--start-maximized')
        self.options.add_argument('--window-size=1920,1080')
        self.options.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=self.options)
        self.wait = WebDriverWait(self.driver, 10)
        
    def __del__(self):
        self.driver.quit()

    def search_amazon_alternatives(self, product_title):
        """
        Search Amazon for alternative products using the given product title.
        Returns a list of alternative products with their prices and links.
        """
        try:
            # Navigate to Amazon
            self.driver.get("https://www.amazon.com")
            
            # Wait for search box and enter query
            search_box = self.wait.until(EC.presence_of_element_located((By.ID, "twotabsearchtextbox")))
            search_box.clear()
            search_box.send_keys(product_title)
            
            # Click search button
            search_button = self.driver.find_element(By.ID, "nav-search-submit-button")
            search_button.click()
            
            # Wait for results to load
            time.sleep(2)
            
            # Get all product elements
            products = self.driver.find_elements(By.XPATH, "//div[@data-component-type='s-search-result']")
            
            alternatives = []
            for product in products[:5]:  # Get top 5 results
                try:
                    title_element = product.find_element(By.XPATH, ".//h2/a")
                    price_element = product.find_element(By.XPATH, ".//span[@class='a-price-whole']")
                    link_element = product.find_element(By.XPATH, ".//h2/a")
                    
                    title = title_element.text.strip()
                    price = price_element.text.strip()
                    link = link_element.get_attribute("href")
                    
                    if title and price and link:
                        alternatives.append({
                            "title": title,
                            "price": price,
                            "link": link
                        })
                except:
                    continue
            
            return alternatives
            
        except Exception as e:
            print(f"Error in Amazon search: {str(e)}")
            return []

def get_cheaper_alternatives(product_title, max_results=5):
    """
    Get alternative products that are cheaper than the original product.
    Uses Selenium to scrape Amazon for alternatives.
    """
    scraper = AlternativeLinkScraper()
    
    try:
        print(f"[DEBUG] Searching Amazon for alternatives...")
        alternatives = scraper.search_amazon_alternatives(product_title)
        
        if alternatives:
            print(f"[DEBUG] Found {len(alternatives)} alternatives on Amazon")
            return alternatives[:max_results]
        else:
            print("[DEBUG] No alternative links found")
            return []
            
    except Exception as e:
        print(f"[DEBUG] Error in get_cheaper_alternatives: {str(e)}")
        return []
    finally:
        del scraper

def extract_base_product_info(product_title):
    """
    Extract base product information from a full product title.
    Returns a tuple of (brand, product_type, simplified_title)
    """
    # Common brand patterns
    brand_patterns = [
        r'(acer|dell|hp|asus|lenovo|apple|samsung|lg|philips|dell|msi|gigabyte|razer|nvidia|intel|amd|google|microsoft|sony|toshiba|huawei|xiaomi|oneplus|htc|nokia|vivo|oppo|realme|motorola|blackberry)'
    ]
    
    # Product type patterns
    product_type_patterns = [
        r'(monitor|laptop|tablet|phone|camera|keyboard|mouse|headphone|speaker|router|ssd|hdd|ram|gpu|cpu|charger|adapter|cable|dock|hub|charger|adapter|headset|earphones|earbuds|charger|adapter|cable|dock|hub)'
    ]
    
    # Price range patterns
    price_patterns = [
        r'\$?\d+(\.\d{2})?',  # Matches prices like $100 or 100.00
        r'\d+gb',  # Matches storage sizes
        r'\d+tb',  # Matches storage sizes
        r'\d+hz',  # Matches refresh rates
        r'\d+gb',  # Matches RAM sizes
        r'\d+mp',  # Matches megapixels
        r'\d+(\.\d+)?"',  # Matches sizes like 24" or 24.5"
    ]
    
    title_lower = product_title.lower()
    
    # Extract brand
    brand = None
    for pattern in brand_patterns:
        match = re.search(pattern, title_lower)
        if match:
            brand = match.group(1)
            break
    
    # Extract product type
    product_type = None
    for pattern in product_type_patterns:
        match = re.search(pattern, title_lower)
        if match:
            product_type = match.group(1)
            break
    
    # Extract price ranges
    price_ranges = []
    for pattern in price_patterns:
        matches = re.findall(pattern, title_lower)
        price_ranges.extend(matches)
    
    # Create simplified titles with various combinations
    simplified_titles = []
    
    # Basic combination: brand + product type
    if brand and product_type:
        simplified_titles.append(f"{brand} {product_type}")
        simplified_titles.append(f"{brand} {product_type} alternative")
        simplified_titles.append(f"{brand} {product_type} similar")
        simplified_titles.append(f"{brand} {product_type} comparison")
    
    # Product type only
    if product_type:
        simplified_titles.extend([
            f"{product_type} alternatives",
            f"{product_type} similar products",
            f"{product_type} comparison",
            f"{product_type} options",
            f"{product_type} deals",
            f"{product_type} choices"
        ])
    
    # Brand only
    if brand:
        simplified_titles.extend([
            f"{brand} products",
            f"{brand} alternatives",
            f"{brand} similar",
            f"{brand} options",
            f"{brand} choices"
        ])
    
    # Add price ranges if any
    if price_ranges:
        for price in price_ranges:
            simplified_titles.extend([
                f"{brand} {product_type} around {price}",
                f"{brand} {product_type} similar to {price}",
                f"{product_type} similar to {price}",
                f"{product_type} around {price}"
            ])
    
    # Add generic queries for common product types
    simplified_titles.extend([
        "best " + product_type if product_type else "",
        "top " + product_type if product_type else "",
        "cheapest " + product_type if product_type else "",
        "most affordable " + product_type if product_type else "",
        "budget " + product_type if product_type else "",
        "value " + product_type if product_type else ""
    ])
    
    # Remove empty strings and duplicates
    simplified_titles = list(filter(None, simplified_titles))
    simplified_titles = list(dict.fromkeys(simplified_titles))
    
    return brand, product_type, price_ranges, simplified_titles

def _test():
    product = "Logitech Wireless Mouse M185"
    brand, product_type, price_ranges, queries = extract_base_product_info(product)
    print(f"Brand: {brand}")
    print(f"Product Type: {product_type}")
    print(f"Price Ranges: {price_ranges}")
    print(f"Simplified Titles: {queries}")
    
    alternatives = get_cheaper_alternatives(product)
    for alt in alternatives:
        print(f"{alt['title']} - ${alt['price']}: {alt['link']}")

if __name__ == "__main__":
    _test()