import streamlit as st

# Set page config for custom title and icon - MUST be the first Streamlit command
st.set_page_config(
    page_title="E-commerce Product Analyzer",
    page_icon="🛒",
    layout="wide"
)

import pandas as pd
import plotly.express as px
import sys
import os
import re
import json
import time
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import base64
from pathlib import Path
import seaborn as sns

# Function to encode the banner image to base64
def get_base64_encoded_image():
    banner_path = Path(__file__).parent.parent / 'images' / 'banner.png'
    with open(banner_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

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

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'stored_link' not in st.session_state:
    st.session_state.stored_link = ''
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'reviews' not in st.session_state:
    st.session_state.reviews = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'loading' not in st.session_state:
    st.session_state.loading = False
if 'product_price' not in st.session_state:
    st.session_state.product_price = None

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

# Navigation functions
def reset_session_state():
    """Reset all processing-related session state variables"""
    st.session_state.analysis_complete = False
    st.session_state.reviews = []
    st.session_state.loading = False

def go_to_input_page():
    st.session_state.page = 'input'
    reset_session_state()

def go_to_results_page():
    st.session_state.page = 'results'

# Page config is already set at the top of the file

# Get the base64 encoded image for the banner
banner_image_base64 = get_base64_encoded_image()

# Custom CSS for banner and input styling
st.markdown(
    f"""
    <style>
    .main-header {{
        background-image: url('data:image/png;base64,{banner_image_base64}');
        background-size: cover;
        background-position: center;
        position: relative;
        color: white;
        padding: 2rem 1rem 1rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0); /* Removed overlay completely to show original image */
        border-radius: 10px;
        z-index: 0;
    }}
    
    .main-header h1, .main-header p {{
        position: relative;
        z-index: 1;
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7); /* Text shadow for better readability */
        font-weight: bold;
    }}
    .stTextInput > div > div > input {{
        background-color: #23272f;
        color: #fff;
        border-radius: 8px;
        border: 1px solid #2a5298;
    }}
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #23272f;
        color: #aaa;
        text-align: center;
        padding: 0.5rem 0;
        font-size: 0.9rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def process_product_link(link):
    """Process the product link and perform all data gathering operations"""
    # Extract product price
    try:
        product_price = extract_price(link)
        st.session_state.product_price = product_price
    except Exception as e:
        st.error(f"Could not extract product price: {e}")
        st.session_state.product_price = None
    
    # Scrape reviews
    try:
        scraper = AmazonReviewScraper(headless=True)
        reviews = scraper.get_reviews(link, max_pages=5, max_reviews=50)
        scraper.close_browser()
    except Exception as e:
        st.error(f"Error scraping reviews: {e}")
        reviews = []
    
    # Store results in session state
    st.session_state.reviews = reviews
    
    # Set loading to False and go to results page
    st.session_state.loading = False
    st.session_state.page = 'results'
    
    # Force a rerun to update the UI
    st.rerun()

def input_page():
    # Banner/Header
    st.markdown(
        '<div class="main-header">'
        '<h1>E-Commerce Product Analyzer</h1>'
        '<p style="font-size:1.5rem;">'
        'Amazon Review Insights & Cheaper Alternatives Finder'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Description with icon
    st.markdown(
        """
        <div style='text-align:center;'>
        <span style='font-size:1.2rem;'>Unlock the Power of Amazon Reviews with AI-Driven Sentiment Analysis.</span>
        <br>
        <span style='font-size:1.2rem;'>Paste a product link below to get started!</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Input section
    link = st.text_input("Enter product link:", key="product_link")
    
    # Center the submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("Analyze Product", use_container_width=True)
    
    if analyze_clicked:
        if link:
            # Store the link and set loading state
            st.session_state.stored_link = link
            st.session_state.loading = True
        else:
            st.error("Please enter a product link first.")
    
    # If loading is True, show loading spinner and process the link
    if st.session_state.loading:
        with st.spinner("Processing your request... This may take a minute."):
            # Process the link
            process_product_link(st.session_state.stored_link)

def results_page():
    # Get data from session state
    link = st.session_state.stored_link
    reviews = st.session_state.reviews
    product_price = st.session_state.product_price
    
    # Banner/Header
    st.markdown(
        '<div class="main-header">'
        '<h1>E-Commerce Product Analyzer</h1>'
        '<p style="font-size:1.5rem;">'
        'Amazon Review Insights & Cheaper Alternatives Finder'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Display a message if no reviews were found
    if not reviews:
        st.warning("No reviews were found for this product.")
        sentiment = "neutral"
        score = 0.5
        pos_count = 0
        neg_count = 0
        detailed_results = []
        model_name = "No reviews available"
    else:
        # Run sentiment analysis on reviews
        sentiment, score, pos_count, neg_count, detailed_results, model_name = predict_sentiment_from_reviews(reviews)
    st.markdown("---")
    st.subheader("Sentiment Analysis Result")
    st.write(f"**Model Used:** {model_name}")
    st.write(f"**Sentiment:** {sentiment.capitalize()} (Score: {score:.2f})")
    st.write(f"**Positive mentions:** {pos_count} | **Negative mentions:** {neg_count}")
    
    if 'avg_confidence' in st.session_state:
        st.write(f"**Average Confidence:** {st.session_state['avg_confidence']}")

    # Display sentiment distribution and summary side by side
    col1, col2 = st.columns(2)
    
    # Display sentiment distribution
    with col1:
        # Create a pie chart of sentiment distribution
        sentiment_counts = {'Positive': pos_count, 'Negative': neg_count}
        fig = px.pie(
            values=list(sentiment_counts.values()),
            names=list(sentiment_counts.keys()),
            title='Sentiment Distribution',
            color_discrete_map={'Positive': '#4b6cb7', 'Negative': '#ff4b4b'},
            color_discrete_sequence=['#ff4b4b', '#4b6cb7'],  # Force color sequence
            hole=0.4
        )
        # Update all text elements to ensure visibility
        fig.update_traces(
            textfont_color='white',
            textfont_size=14,
            textposition='inside',
            insidetextorientation='horizontal'
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#333333',
            title_font_color='#333333',
            title_font_size=16,
            legend=dict(
                font=dict(color='#333333', size=12),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#dddddd'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display sentiment summary
    with col2:
        st.markdown(f"**Total Reviews Analyzed:** {len(reviews)}")
        st.markdown(f"**Positive Reviews:** {pos_count} ({pos_count/len(reviews)*100 if reviews else 0:.1f}%)")
        st.markdown(f"**Negative Reviews:** {neg_count} ({neg_count/len(reviews)*100 if reviews else 0:.1f}%)")
        
        # Display overall sentiment
        st.markdown("### Overall Sentiment")
        if sentiment == "positive":
            st.markdown("<div style='background-color:#4b6cb7; color:white; padding:10px; border-radius:5px;'>Positive</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background-color:#ff4b4b; color:white; padding:10px; border-radius:5px;'>Negative</div>", unsafe_allow_html=True)

    # Show detailed sentiment analysis
    st.subheader("Detailed Sentiment Analysis")
    
    # Create two columns for the charts
    col1, col2 = st.columns(2)
    
    # Show ratings distribution (original ratings 1-5 stars)
    with col1:
        if 'sentiment_score' in st.session_state:
            st.plotly_chart(st.session_state['sentiment_score'], use_container_width=True)
    
    # Show sentiment distribution (positive/negative)
    with col2:
        if 'sentiment_distribution' in st.session_state:
            st.plotly_chart(st.session_state['sentiment_distribution'], use_container_width=True)
    
    # Product recommendations using SerpApi
    st.markdown("---")
    st.subheader("Product Recommendations")
    
    # Determine how many products to display based on sentiment
    max_exact = 10  # Request more results to have more options for filtering
    max_alternatives = 10
    
    if sentiment == "positive":
        st.success(f"This product is recommended! Here's the best value for this product:")
        display_max = 1  # Only show 1 result for positive sentiment
    else:
        st.warning(f"This product may not be ideal. Consider these alternatives:")
        display_max = 5  # Show up to 5 results for negative sentiment
        
    # Get product title from the first review or fallback
    raw_title = reviews[0].get('product_title', '') if reviews and 'product_title' in reviews[0] else "Amazon Product"
    
    # Simplify the product title (take first part before colon, dash, or parenthesis)
    match = re.split(r'[:\-\(]', raw_title)
    simple_title = match[0].strip() if match else raw_title.strip()
    if not simple_title:
        simple_title = "Product Name"
        
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
        else:
            # If no exact matches, display a message
            st.warning("No exact product matches found. Please try a different product.")
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
        else:
            # If no alternatives, display a message
            st.warning("No alternative products found. Please try a different product.")
    
    # If we still don't have any products to display, show a message
    if not display_products:
        st.error("No products available to display. Please try a different product or search term.")
        
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
        # Map the actual column names from the DataFrame to display names
        column_mapping = {
            'title': 'Product',
            'price': 'Price',
            'source': 'Source',
            'rating': 'Rating',
            'reviews': 'Reviews',
            'link': 'Link'
        }
        
        # Rename the columns to match the display names
        df = df.rename(columns=column_mapping)
        
        # Select columns for display
        display_cols = ['Product', 'Price', 'Source', 'Rating', 'Reviews', 'Link']
        display_cols = [col for col in display_cols if col in df.columns]
        
        # Display the table with HTML formatting and improved styling
        styled_table = df[display_cols].to_html(escape=False, index=False)
        
        # Apply custom CSS to make the table more visible and centered
        styled_table = f"""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <div style="width: 95%; max-width: 1000px;">
                <style>
                    table {{width: 100%; border-collapse: collapse; margin: 25px 0; font-size: 16px; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);}}
                    th {{background-color: #4b6cb7; color: white; text-align: center; padding: 12px 15px; font-weight: bold;}}
                    td {{padding: 12px 15px; text-align: center; border-bottom: 1px solid #dddddd;}}
                    tr:nth-child(even) {{background-color: #f8f8f8;}}
                    tr:hover {{background-color: #f1f1f1;}}
                    a {{color: #4b6cb7; text-decoration: none; font-weight: bold;}}
                    a:hover {{text-decoration: underline;}}
                </style>
                {styled_table}
            </div>
        </div>
        """
        
        st.markdown(styled_table, unsafe_allow_html=True)
        
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
    
    # Try Again button at the bottom and centered
    st.markdown("---")
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Try Another Product", key="try_again_bottom", use_container_width=True):
            # Clear all session state variables
            for key in list(st.session_state.keys()):
                if key != 'page':  # Keep the page key
                    del st.session_state[key]
            # Set page to input
            st.session_state.page = 'input'
            # Force rerun to refresh the page immediately
            st.rerun()

# Main function to manage page navigation
def main():
    # Apply custom CSS for white/blueish theme with light gray background
    st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5;
        }
        .stApp {
            background-color: #f5f5f5;
        }
        .main-header {
            /* Removed background gradient to allow the image to show */
            color: white;
            padding: 2rem 1rem 1rem 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stTextInput > div > div > input {
            background-color: #ffffff;
            color: #212529;
            border-radius: 8px;
            border: 1px solid #4b6cb7;
        }
        .stButton > button {
            background-color: #4b6cb7;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
            width: 50% !important;
            margin: 0 auto;
            display: block;
        }
        .stButton > button:hover {
            background-color: #3a5795;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: #f0f0f0;
            color: #6c757d;
            text-align: center;
            padding: 0.5rem 0;
            font-size: 0.9rem;
            border-top: 1px solid #e9ecef;
        }
        .stSpinner > div > div > div {
            border-color: #4b6cb7 #4b6cb7 transparent !important;
        }
        .stProgress > div > div > div {
            background-color: #4b6cb7 !important;
        }
        .stAlert > div {
            border-radius: 8px;
            border-left-color: #4b6cb7 !important;
        }
        h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown {
            color: #333333 !important;
        }
        .stDataFrame {
            color: #333333;
        }
        .stTable {
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Display the appropriate page based on session state
    if st.session_state.page == 'input':
        input_page()
    else:
        results_page()
    
    # Footer
    st.markdown(
        '<div class="footer">'
        'Amazon E-Commerce Product Analyzer &copy; 2025'
        '</div>',
        unsafe_allow_html=True
    )

# Run the main function
if __name__ == "__main__":
    main()