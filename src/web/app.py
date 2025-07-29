import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import re

# Add the project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) 

# Import from the new directory structure using absolute imports
from src.scraper.amazon_review_scraper import AmazonReviewScraper
from src.models.model_integration import SentimentAnalyzer
from src.api.serp_api_integration import get_product_alternatives

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
    
    # Show detailed results with confidence scores
    st.subheader("🔍 Detailed Review Analysis")
    for i, result in enumerate(detailed_results):
        sentiment_emoji = ':smile:' if result['sentiment'] == 'positive' else ':disappointed:'
        strength = result.get('sentiment_strength', 'moderate')
        strength_color = {
            'strong': 'green' if result['sentiment'] == 'positive' else 'red',
            'moderate': 'orange',
            'weak': 'gray'
        }.get(strength, 'blue')
        
        st.markdown(f"**Review {i+1}:** {sentiment_emoji} {result['sentiment'].capitalize()} ")
        st.markdown(f"**Confidence:** <span style='color:{strength_color}'>{result['confidence']:.2f} ({strength})</span>", unsafe_allow_html=True)
        st.write(f"Rating: {result['rating']}")
        st.write(f"Helpful Votes: {result['helpful_votes']}")
        st.write(f"Review Text: {result['review']}")
        st.markdown("---")

    # Product recommendations using SerpApi
    st.markdown("---")
    st.subheader("\U0001F4A1 Product Recommendations")
    if sentiment == "positive":
        st.success("This product is recommended! Here are some similar options:")
    else:
        st.warning("This product may not be ideal. Consider these alternatives:")
        
    # Get product title from the first review or fallback
    raw_title = reviews[0].get('product_title', '') if reviews and 'product_title' in reviews[0] else "Amazon Product"
    
    # Simplify the product title (take first part before colon, dash, or parenthesis)
    match = re.split(r'[:\-\(]', raw_title)
    simple_title = match[0].strip() if match else raw_title.strip()
    if not simple_title:
        simple_title = "Apple iPad"
        
    # Get product alternatives using SerpApi
    alternatives = get_product_alternatives(simple_title)
    
    # Fallback: try a generic query if no results
    if not alternatives:
        fallback_title = "Apple iPad"
        st.write(f"No results found. Trying with fallback title: {fallback_title}")
        alternatives = get_product_alternatives(fallback_title)
        
    # Display results
    if alternatives:
        df = pd.DataFrame(alternatives)
        # Format the link column as markdown links
        df['link'] = df['link'].apply(lambda x: f"[Link]({x})")
        # Select and reorder columns for display
        display_cols = ['title', 'price', 'source', 'rating', 'reviews', 'link']
        display_cols = [col for col in display_cols if col in df.columns]
        st.write(df[display_cols].to_markdown(index=False), unsafe_allow_html=True)
    else:
        st.info("No alternative products found.")

st.markdown("---")

# Sidebar Documentation
st.sidebar.title("📖 Documentation")
st.sidebar.markdown("""
**User Guide:**
- Enter a product link in the input box.
- The app will analyze the product's sentiment (dummy model for now).
- Based on the result, you'll see either cheaper options or alternatives.
- Results are mock data; integrate real APIs and models as needed.

**Developer Guide:**
- Replace `predict_sentiment` with your ML model.
- Replace `search_ecommerce` with real e-commerce API calls.
- Customize UI and visualizations as needed.
""")

# Footer
st.markdown(
    '<div class="footer">'
    'Capstone 2 &copy; 2025 &mdash; Developed by Umar'
    '</div>',
    unsafe_allow_html=True
) 