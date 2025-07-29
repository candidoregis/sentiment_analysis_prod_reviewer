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

You can provide your credentials in two ways:

### Option 1: Environment Variables (Recommended)

Create a `.env` file in the project root directory with the following variables:

```
# Amazon credentials
AMAZON_EMAIL=your-email@example.com
AMAZON_PASSWORD=your-password

# SerpApi credentials
SERPAPI_KEY=your-serpapi-key
```

A template file `.env.template` is provided for reference. Copy this file to `.env` and add your credentials:

```bash
cp .env.template .env
```

### Option 2: JSON Configuration File

Alternatively, you can place your Amazon credentials in `config/credentials.json` in the following format:

```json
{
  "amazon": {
    "email": "your-email@example.com",
    "password": "your-password"
  },
  "serpapi": {
    "key": "your-serpapi-key"
  }
}
```

**Note:** Environment variables take precedence over the JSON configuration file.

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
