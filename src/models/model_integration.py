import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

class SentimentAnalyzer:
    def __init__(self):
        # Load the trained model and related objects
        import os
        model_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            print("Loading sentiment analysis model...")
            self.model = joblib.load(os.path.join(model_dir, 'sentiment_model.pkl'))
            self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
            print("MLP (Imbalanced) model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback model...")
            # If the model files don't exist, create a simple fallback model
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression()
            self.vectorizer = None
            self.label_encoder = None
        
    def analyze_reviews(self, reviews):
        """
        Analyze a list of review texts and return sentiment analysis results
        """
        if not reviews:
            return {
                'overall_sentiment': 'neutral',
                'score': 0.5,
                'positive_count': 0,
                'negative_count': 0,
                'detailed_results': [],
                'model_name': 'MLP (Imbalanced)'
            }
        
        if self.vectorizer is None or self.label_encoder is None:
            # Fallback mode - simple sentiment analysis
            return self._fallback_analysis(reviews)
            
        try:
            # Transform and predict each review using our trained model
            texts = [review.get('body', '') for review in reviews]
            X_tfidf = self.vectorizer.transform(texts)
            predictions = self.model.predict(X_tfidf)
            probabilities = self.model.predict_proba(X_tfidf)
            
            # Convert predictions back to labels
            try:
                # Try to directly map the predictions to labels without using the encoder
                # This avoids the case sensitivity issue completely
                sentiments = ['positive' if p == 1 else 'negative' for p in predictions]
                print(f"Successfully mapped predictions to labels directly")
            except Exception as e:
                print(f"Error in direct label mapping: {e}")
                try:
                    # Fall back to using the encoder if direct mapping fails
                    sentiments = self.label_encoder.inverse_transform(predictions)
                    # Convert all labels to lowercase for consistency
                    sentiments = [s.lower() if isinstance(s, str) else s for s in sentiments]
                except Exception as e:
                    print(f"Error transforming labels with encoder: {e}")
                    # Last resort fallback
                    sentiments = ['positive' if p == 1 else 'negative' for p in predictions]
            
            # Count positive and negative reviews - ensure case-insensitive comparison
            positive_count = sum(1 for s in sentiments if str(s).lower() == 'positive')
            negative_count = sum(1 for s in sentiments if str(s).lower() == 'negative')
            total = positive_count + negative_count
            
            # Calculate overall sentiment score
            score = positive_count / total if total > 0 else 0.5
            overall_sentiment = 'positive' if score >= 0.5 else 'negative'
            
            # Calculate confidence levels
            avg_confidence = float(sum(prob.max() for prob in probabilities) / len(probabilities)) if probabilities.size > 0 else 0.5
            
            # Prepare detailed results
            detailed_results = []
            for i, (review, sentiment, prob) in enumerate(zip(reviews, sentiments, probabilities)):
                confidence = prob.max()
                sentiment_strength = 'strong' if confidence > 0.8 else 'moderate' if confidence > 0.6 else 'weak'
                
                detailed_results.append({
                    'review': review.get('body', ''),
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'sentiment_strength': sentiment_strength,
                    'rating': review.get('rating', None),
                    'helpful_votes': review.get('helpful_votes', 0)
                })
                
            # Sort by confidence (most confident first)
            detailed_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'overall_sentiment': overall_sentiment,
                'score': score,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'average_confidence': avg_confidence,
                'detailed_results': detailed_results,
                'model_name': 'MLP (Imbalanced)'
            }
            
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return self._fallback_analysis(reviews)
            
    def _fallback_analysis(self, reviews):
        """Simple fallback sentiment analysis when model loading fails"""
        import re
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 'perfect']
        negative_words = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'disappointing', 'broken']
        
        detailed_results = []
        positive_count = 0
        negative_count = 0
        
        for review in reviews:
            text = review.get('body', '').lower()
            pos_matches = sum(1 for word in positive_words if re.search(r'\b' + word + r'\b', text))
            neg_matches = sum(1 for word in negative_words if re.search(r'\b' + word + r'\b', text))
            
            sentiment = 'positive' if pos_matches > neg_matches else 'negative'
            confidence = 0.5 + (abs(pos_matches - neg_matches) / 10)
            confidence = min(confidence, 0.9)  # Cap at 0.9 for fallback mode
            
            if sentiment == 'positive':
                positive_count += 1
            else:
                negative_count += 1
                
            detailed_results.append({
                'review': review.get('body', ''),
                'sentiment': sentiment,
                'confidence': float(confidence),
                'sentiment_strength': 'moderate',
                'rating': review.get('rating', None),
                'helpful_votes': review.get('helpful_votes', 0)
            })
            
        total = positive_count + negative_count
        score = positive_count / total if total > 0 else 0.5
        overall_sentiment = 'positive' if score >= 0.5 else 'negative'
        
        return {
            'overall_sentiment': overall_sentiment,
            'score': score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'detailed_results': detailed_results,
            'model_name': 'Fallback Model'
        }
        
    def create_visualizations(self, analysis_results):
        """
        Create visualizations for the sentiment analysis results
        """
        # Create sentiment distribution plot
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=['Positive', 'Negative'],
            y=[analysis_results['positive_count'], analysis_results['negative_count']],
            marker_color=['#1e90ff', '#ff4b4b']
        ))
        fig1.update_layout(
            title=f'Sentiment Distribution ({analysis_results.get("model_name", "Model")})',
            xaxis_title='Sentiment',
            yaxis_title='Number of Reviews',
            plot_bgcolor='#23272f',
            paper_bgcolor='#23272f',
            font_color='#fff'
        )
        
        # Create sentiment score gauge
        fig2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = analysis_results['score'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1e90ff" if analysis_results['score'] >= 0.5 else "#ff4b4b"},
                'steps' : [
                    {'range': [0, 50], 'color': "#ff4b4b"},
                    {'range': [50, 100], 'color': "#1e90ff"}
                ]
            }
        ))
        fig2.update_layout(
            plot_bgcolor='#23272f',
            paper_bgcolor='#23272f',
            font_color='#fff'
        )
        
        return fig1, fig2

# Example usage
def main():
    analyzer = SentimentAnalyzer()
    
    # Example reviews (replace with actual reviews)
    example_reviews = [
        {'body': 'This product is amazing! It works perfectly.', 'rating': 5},
        {'body': 'I am disappointed with this product.', 'rating': 2},
        {'body': 'Great value for money!', 'rating': 4}
    ]
    
    results = analyzer.analyze_reviews(example_reviews)
    print("\nAnalysis Results:")
    print(f"Overall Sentiment: {results['overall_sentiment']}")
    print(f"Score: {results['score']:.2f}")
    print(f"Positive Reviews: {results['positive_count']}")
    print(f"Negative Reviews: {results['negative_count']}")
    
    # Create visualizations
    fig1, fig2 = analyzer.create_visualizations(results)
    fig1.show()
    fig2.show()

if __name__ == "__main__":
    main()
