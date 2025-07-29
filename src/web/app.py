import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import re
import time
import random
from datetime import datetime

# Add the project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Create output directory for saved data if it doesn't exist
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/data'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import from the new directory structure using absolute imports
from src.scraper.amazon_review_scraper import AmazonReviewScraper
from src.scraper.amazon_price_extractor import extract_price
from src.models.model_integration import SentimentAnalyzer
from src.api.serp_api_integration import get_exact_and_alternative_products

# Initialize the sentiment analyzer
analyzer = SentimentAnalyzer()

def predict_sentiment_from_reviews(reviews):
    """
    Analyze reviews using the trained model
    """
    results = analyzer.analyze_reviews(reviews)
    sentiment = results['overall_sentiment']
    score = results['score']
    pos_count = results['positive_count']
    neg_count = results['negative_count']
    model_name = results.get('model_name', 'Sentiment Model')
    avg_confidence = results.get('average_confidence', None)
    
    # Create visualizations
    fig1, fig2 = analyzer.create_visualizations(results)
    
    # Store visualizations in session state
    st.session_state['sentiment_distribution'] = fig1
    st.session_state['sentiment_score'] = fig2
    st.session_state['model_name'] = model_name
    
    if avg_confidence:
        st.session_state['avg_confidence'] = f"{avg_confidence:.2f}"
    
    return sentiment, score, pos_count, neg_count, results['detailed_results'], model_name

# E-commerce search function using SerpApi
def search_ecommerce(product_link, sentiment):
    # This function is kept for backward compatibility
    # The actual implementation is now in serp_api_integration.py
    return []

# Set page config for custom title and icon
st.set_page_config(
    page_title="E-commerce Product Analyzer",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS for banner and input styling
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem 1rem 1rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .stTextInput > div > div > input {
        background-color: #23272f;
        color: #fff;
        border-radius: 8px;
        border: 1px solid #2a5298;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #23272f;
        color: #aaa;
        text-align: center;
        padding: 0.5rem 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Banner/Header
st.markdown(
    '<div class="main-header">'
    '<h1>🛒 E-commerce Product Analyzer</h1>'
    '<p style="font-size:1.2rem;">'
    'Analyze product sentiment and discover the best shopping options!'
    '</p>'
    '</div>',
    unsafe_allow_html=True
)

# Description with icon
st.markdown(
    """
    <div style='text-align:center;'>
    <span style='font-size:1.5rem;'>🔗 Paste a product link below to get started!</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Input section
link = st.text_input("Enter product link:", key="product_link")

if link:
    # Extract product price first (silently, without displaying)
    try:
        product_price = extract_price(link)
    except Exception as e:
        print(f"Could not extract product price: {e}")
        product_price = None
    
    # Then scrape reviews
    with st.spinner("Scraping Amazon reviews (up to 5 pages)..."):
        try:
            scraper = AmazonReviewScraper(headless=True)
            reviews = scraper.get_reviews(link, max_pages=5, max_reviews=50)
            scraper.close_browser()
        except Exception as e:
            st.error(f"Error scraping reviews: {e}")
            reviews = []
    if reviews:
        st.markdown("---")
        st.subheader("📝 Sample Reviews")
        for i, review in enumerate(reviews[:5]):
            st.write(f"**Review {i+1}:** {review.get('body', '')}")
        # Run sentiment analysis on reviews
        sentiment, score, pos_count, neg_count, detailed_results, model_name = predict_sentiment_from_reviews(reviews)
    else:
        st.warning("No reviews found or failed to scrape reviews.")
        sentiment, score, pos_count, neg_count, detailed_results, model_name = "neutral", 0.5, 0, 0, [], "No Model"
    st.markdown("---")
    st.subheader("📊 Sentiment Analysis Result")
    st.write(f"**Model Used:** {model_name}")
    st.write(f"**Sentiment:** :{'smile:' if sentiment=='positive' else 'disappointed:'} {sentiment.capitalize()} (Score: {score:.2f})")
    st.write(f"**Positive mentions:** {pos_count} | **Negative mentions:** {neg_count}")
    
    if 'avg_confidence' in st.session_state:
        st.write(f"**Average Confidence:** {st.session_state['avg_confidence']}")

    # Show detailed sentiment analysis
    st.subheader("📊 Detailed Sentiment Analysis")
    
    # Show sentiment distribution
    if 'sentiment_distribution' in st.session_state:
        st.plotly_chart(st.session_state['sentiment_distribution'], use_container_width=True)
    
    # Show sentiment score gauge
    if 'sentiment_score' in st.session_state:
        st.plotly_chart(st.session_state['sentiment_score'], use_container_width=True)
    
    # Product recommendations using SerpApi
    st.markdown("---")
    st.subheader("\U0001F4A1 Product Recommendations")
    
    # Determine how many products to display based on sentiment
    max_exact = 10  # Request more results to have more options for filtering
    max_alternatives = 10
    
    if sentiment == "positive":
        st.success(f"This product is recommended! Here's the best value similar option:")
        display_max = 1  # Only show 1 result for positive sentiment
    else:
        st.warning(f"This product may not be ideal. Consider these alternatives:")
        display_max = 5  # Show up to 5 results for negative sentiment
        
    # Get product title from the first review or fallback
    raw_title = reviews[0].get('product_title', '') if reviews and 'product_title' in reviews[0] else "Amazon Product"
    
    # Note: product_price is already extracted earlier in the code
    # We don't need to extract it from reviews anymore
    
    # Simplify the product title (take first part before colon, dash, or parenthesis)
    match = re.split(r'[:\-\(]', raw_title)
    simple_title = match[0].strip() if match else raw_title.strip()
    if not simple_title:
        simple_title = "Apple iPad"
        
    # Get both exact matches and alternatives using our new function
    with st.spinner("Searching for product alternatives..."):
        exact_matches, alternatives = get_exact_and_alternative_products(simple_title, max_exact=max_exact, max_alternatives=max_alternatives)
        
        # No CSV saving here - will save after processing
    
    # Fallback: try a generic query if no results
    if not exact_matches and not alternatives:
        fallback_title = simple_title.split()[0] if simple_title else "product"
        st.write(f"No results found. Trying with fallback title: {fallback_title}")
        exact_matches, alternatives = get_exact_and_alternative_products(
            fallback_title, max_exact=max_exact, max_alternatives=max_alternatives
        )
    
    # Helper function to convert price strings to float
    def extract_price_float(product):
        if 'price' in product and product['price']:
            try:
                # Clean the price string and convert to float
                price_str = product['price'].replace('$', '').replace(',', '').strip()
                return float(price_str)
            except (ValueError, TypeError):
                return None
        return None
    
    # Process exact matches - convert prices to float
    for product in exact_matches:
        product['price_float'] = extract_price_float(product)
    
    # Process alternatives - convert prices to float
    for product in alternatives:
        product['price_float'] = extract_price_float(product)
    
    # Final products to display
    display_products = []
    
    # Apply different filtering logic based on sentiment
    if sentiment == "positive":
        # For positive sentiment: show the cheapest exact match
        if exact_matches and product_price is not None:
            # Filter exact matches with valid prices
            valid_price_products = [p for p in exact_matches if p.get('price_float') is not None]
            
            if valid_price_products:
                # Sort by price (cheapest first)
                cheapest_products = sorted(valid_price_products, key=lambda x: x['price_float'])
                # Take the cheapest one
                display_products = [cheapest_products[0]]
                st.info(f"Showing the cheapest exact match: ${cheapest_products[0]['price_float']:.2f}")
            else:
                # If no products with valid prices, just take the first one
                display_products = [exact_matches[0]]
        elif exact_matches:
            # If we have exact matches but no product price, just take the first one
            display_products = [exact_matches[0]]
        elif alternatives:
            # If no exact matches but we have alternatives, take the first alternative
            display_products = [alternatives[0]]
    else:  # negative sentiment
        # For negative sentiment: select 5 items from alternatives with prices closest to original
        if alternatives and product_price is not None:
            # Filter alternatives with valid price information
            valid_price_alternatives = [alt for alt in alternatives if alt.get('price_float') is not None]
            
            if valid_price_alternatives:
                # Calculate price difference from original product for each alternative
                for alt in valid_price_alternatives:
                    alt['price_diff'] = abs(alt['price_float'] - product_price)
                
                # Sort alternatives by price difference (closest first)
                closest_price_alternatives = sorted(valid_price_alternatives, key=lambda x: x['price_diff'])
                
                # Take the 5 closest matches
                display_products = closest_price_alternatives[:display_max]
                
                st.info(f"Showing {len(display_products)} alternatives with prices closest to the original (${product_price:.2f}).")
            else:
                # If no alternatives with valid prices, just take the first few alternatives
                display_products = alternatives[:display_max]
                st.info(f"Showing alternatives (price comparison not available).")
        elif alternatives:
            # If we have alternatives but no product price, just take the first few
            display_products = alternatives[:display_max]
        elif exact_matches:
            # If no alternatives but we have exact matches, use those
            display_products = exact_matches[:display_max]
    
    # If we still don't have any products to display, use a combination of both lists
    if not display_products:
        combined = exact_matches + alternatives
        display_products = combined[:display_max]
        
    # Display results
    if display_products:
        df = pd.DataFrame(display_products)
        
        # Handle missing links
        for i, product in enumerate(display_products):
            if 'link' not in product or not product['link']:
                # If no link, indicate that no link is available
                product['link'] = "No link available"
                df.at[i, 'link'] = product['link']
        
        # Format the link column as clickable markdown links only for valid links
        if 'link' in df.columns:
            df['link'] = df['link'].apply(lambda x: f"[View Product]({x})" if x and x != "No link available" else "No link available")
        
        # Select and reorder columns for display
        display_cols = ['title', 'price', 'source', 'rating', 'reviews', 'link']
        display_cols = [col for col in display_cols if col in df.columns]
        
        # Display the table with HTML formatting
        st.markdown(df[display_cols].to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # After displaying the data, save it to CSV files
        # Extract product ID from URL or use a random number
        product_id = re.search(r'/dp/([A-Z0-9]{10})/', link)
        if product_id:
            product_id = product_id.group(1)
        else:
            product_id = f"{random.randint(1000, 9999)}"
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save reviews to CSV
        if reviews:
            reviews_df = pd.DataFrame(reviews)
            reviews_filename = f"product_{product_id}_{timestamp}.csv"
            reviews_filepath = os.path.join(OUTPUT_DIR, reviews_filename)
            reviews_df.to_csv(reviews_filepath, index=False)
            print(f"Saved {len(reviews)} reviews to {reviews_filepath}")
        
        # Save exact matches to CSV
        if exact_matches:
            exact_df = pd.DataFrame(exact_matches)
            exact_filename = f"product_{product_id}_{timestamp}_exact_match.csv"
            exact_filepath = os.path.join(OUTPUT_DIR, exact_filename)
            exact_df.to_csv(exact_filepath, index=False)
            print(f"Saved {len(exact_matches)} exact matches to {exact_filepath}")
        
        # Save alternatives to CSV
        if alternatives:
            alt_df = pd.DataFrame(alternatives)
            alt_filename = f"product_{product_id}_{timestamp}_alternatives.csv"
            alt_filepath = os.path.join(OUTPUT_DIR, alt_filename)
            alt_df.to_csv(alt_filepath, index=False)
            print(f"Saved {len(alternatives)} alternatives to {alt_filepath}")

    else:
        st.info("No alternative products found.")

st.markdown("---")

# Sidebar - removed documentation

# Footer
st.markdown(
    '<div class="footer">'
    'Amazon E-Commerce Product Analyzer &copy; 2025 &mdash;'
    '</div>',
    unsafe_allow_html=True
) 