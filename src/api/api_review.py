import requests
import json
from typing import List, Dict, Optional

class APIExplorer:
    def __init__(self):
        self.api_keys = {
            'google_sentiment': None,  # Add your Google Cloud API key
            'amazon_product': None,    # Add your Amazon Product Advertising API key
            'senticnet': None         # Add your SenticNet API key if needed
        }
        
    def test_google_sentiment(self, text: str) -> Dict:
        """
        Test Google Cloud Natural Language API for sentiment analysis
        """
        if not self.api_keys['google_sentiment']:
            return {'error': 'Google API key not configured'}
            
        url = "https://language.googleapis.com/v1/documents:analyzeSentiment"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_keys['google_sentiment']}'
        }
        data = {
            'document': {
                'type': 'PLAIN_TEXT',
                'content': text
            },
            'encodingType': 'UTF8'
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()
        except Exception as e:
            return {'error': str(e)}
            
    def test_amazon_product_search(self, product_id: str) -> Dict:
        """
        Test Amazon Product Advertising API
        """
        if not self.api_keys['amazon_product']:
            return {'error': 'Amazon API key not configured'}
            
        params = {
            'Operation': 'ItemLookup',
            'ItemId': product_id,
            'ResponseGroup': 'Large',
            'AWSAccessKeyId': self.api_keys['amazon_product']
        }
        
        try:
            response = requests.get('https://webservices.amazon.com/onca/xml', params=params)
            return {'xml': response.text}
        except Exception as e:
            return {'error': str(e)}
            
    def test_senticnet(self, text: str) -> Dict:
        """
        Test SenticNet API for sentiment analysis
        """
        if not self.api_keys['senticnet']:
            return {'error': 'SenticNet API key not configured'}
            
        url = "https://api.sentic.net/v1/analyze"
        headers = {
            'Authorization': f'Bearer {self.api_keys['senticnet']}'
        }
        
        try:
            response = requests.post(url, headers=headers, json={'text': text})
            return response.json()
        except Exception as e:
            return {'error': str(e)}
            
    def compare_apis(self, sample_reviews: List[str]) -> Dict:
        """
        Compare different APIs on sample data
        """
        results = {}
        
        for api_name in ['google_sentiment', 'senticnet']:
            api_method = getattr(self, f'test_{api_name}')
            api_results = []
            
            for review in sample_reviews:
                result = api_method(review)
                api_results.append({
                    'review': review,
                    'result': result
                })
            
            results[api_name] = {
                'name': api_name,
                'results': api_results
            }
        
        return results

def main():
    print("\nTesting alternative APIs for sentiment analysis...")
    
    # Sample reviews for testing
    sample_reviews = [
        "This product is amazing! It works perfectly.",
        "I'm disappointed with this product. It broke after one use.",
        "The quality is terrible and it doesn't work as advertised.",
        "Great value for money. Highly recommend this product!",
        "It's okay, but nothing special."
    ]
    
    # Initialize API explorer
    explorer = APIExplorer()
    
    # Compare APIs
    results = explorer.compare_apis(sample_reviews)
    
    # Save results to JSON for documentation
    with open('api_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nAPI comparison results saved to api_comparison.json")
    print("\nNote: Some APIs may require API keys to be configured")

if __name__ == "__main__":
    main()
