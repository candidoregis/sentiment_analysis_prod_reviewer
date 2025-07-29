import joblib
import numpy as np

def test_model():
    print("\nTesting saved model...")
    
    try:
        # Load the saved model files
        model = joblib.load('models/sentiment_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        print("\nModel files loaded successfully!")
        
        # Test the model with some example reviews
        test_reviews = [
            "This product is amazing! It works perfectly and I love it.",
            "I'm disappointed with this product. It broke after one use.",
            "The quality is terrible and it doesn't work as advertised.",
            "Great value for money. Highly recommend this product!",
            "It's okay, but nothing special."
        ]
        
        print("\nTesting with sample reviews:")
        for review in test_reviews:
            # Transform the text
            X_test = vectorizer.transform([review])
            
            # Predict sentiment
            prediction = model.predict(X_test)[0]
            confidence = model.predict_proba(X_test).max()
            
            # Convert prediction back to label
            sentiment = label_encoder.inverse_transform([prediction])[0]
            
            print(f"\nReview: {review}")
            print(f"Predicted Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.2f}")
            
    except Exception as e:
        print(f"\nError testing model: {str(e)}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_model()
