from model_integration import SentimentAnalyzer
import json

def test_model_integration():
    print("Testing model integration...")
    
    # Initialize the sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Test reviews
    test_reviews = [
        "This product is amazing! It works perfectly.",
        "I'm disappointed with this product. It broke after one use.",
        "The quality is terrible and it doesn't work as advertised.",
        "Great value for money. Highly recommend this product!",
        "It's okay, but nothing special."
    ]
    
    # Analyze reviews
    results = analyzer.analyze_reviews(test_reviews)
    
    print("\nAnalysis Results:")
    print(f"Overall Sentiment: {results['overall_sentiment']}")
    print(f"Sentiment Score: {results['score']:.2f}")
    print(f"Positive Reviews: {results['positive_count']}")
    print(f"Negative Reviews: {results['negative_count']}")
    
    print("\nDetailed Results:")
    for i, review in enumerate(test_reviews):
        result = results['detailed_results'][i]
        print(f"\nReview {i+1}: {review}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
    
    # Save visualizations to files
    fig1, fig2 = analyzer.create_visualizations(results)
    fig1.write_image("sentiment_distribution.png")
    fig2.write_image("sentiment_score.png")
    
    print("\nVisualizations saved to sentiment_distribution.png and sentiment_score.png")
    
    # Save results to JSON for verification
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to test_results.json")

if __name__ == "__main__":
    test_model_integration()
