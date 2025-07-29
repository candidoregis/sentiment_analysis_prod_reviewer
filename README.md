# Amazon Review Scraper and Sentiment Analysis

A comprehensive tool for scraping Amazon product reviews and performing sentiment analysis on them. The application includes a web interface for easy interaction.

## Project Structure

```
amazon-scrape/
├── config/                  # Configuration files
│   ├── amazon_cookies.json
│   ├── credentials.json
│   └── debug_info.json
├── data/                    # Data storage
│   ├── processed/           # Processed data files
│   └── raw/                 # Raw data files (Reviews.csv)
├── src/                     # Source code
│   ├── api/                 # API endpoints
│   │   └── api_review.py
│   ├── models/              # Sentiment analysis models
│   │   ├── compare_models.py
│   │   ├── debug_sentiment.py
│   │   ├── model_integration.py
│   │   ├── sentiment_analysis.py
│   │   ├── train_model.py
│   │   ├── label_encoder.pkl
│   │   ├── sentiment_model.pkl
│   │   └── vectorizer.pkl
│   ├── scraper/             # Amazon review scraping functionality
│   │   ├── amazon_review_scraper.py
│   │   ├── interactive_review_scraper.py
│   │   └── review_scraper.py
│   ├── tests/               # Test files
│   │   ├── test_csv.py
│   │   ├── test_integration.py
│   │   ├── test_model.py
│   │   └── test_serpapi.py
│   ├── utils/               # Utility functions
│   │   ├── check_data.py
│   │   ├── create_sample.py
│   │   ├── credentials_manager.py
│   │   └── selenium_integration.py
│   └── web/                 # Web interface
│       ├── app.py
│       └── deployment.py
├── static/                  # Static assets (HTML, images)
├── main.py                  # Main entry point
└── requirements.txt         # Project dependencies
```

## Installation

1. Clone the repository
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Interface

```
python main.py
```

### Using the Scraper Directly

```python
from src.scraper.amazon_review_scraper import AmazonReviewScraper

scraper = AmazonReviewScraper(headless=True)
reviews = scraper.get_reviews("https://www.amazon.com/product-url", max_pages=5)
scraper.close_browser()
```

### Running Sentiment Analysis

```python
from src.models.model_integration import SentimentAnalyzer

analyzer = SentimentAnalyzer()
results = analyzer.analyze_reviews(reviews)
```

## Configuration

Place your Amazon credentials in `config/credentials.json` in the following format:

```json
{
  "amazon": {
    "email": "your-email@example.com",
    "password": "your-password"
  }
}
```

## Dependencies

- selenium
- webdriver-manager
- pandas
- beautifulsoup4
- flask
- scikit-learn
- joblib
- matplotlib
- seaborn
- plotly
- google-search-results

## License

This project is licensed under the MIT License.
