import pandas as pd
import numpy as np
import os
import csv
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
from tabulate import tabulate

# For version 3 - Advanced MLP model with TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def get_project_root():
    """Get the absolute path to the project root directory"""
    # This assumes the file is in src/models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../..'))
    return project_root

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_and_preprocess_data(sample_size=50000):
    """Load and preprocess the dataset for sentiment analysis"""
    print("Loading dataset...")
    # Use relative path from project root
    data_path = os.path.join(get_project_root(), 'data', 'raw', 'Reviews.csv')
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find the data file at {data_path}")
        print("Please make sure the Reviews.csv file exists in the data/raw directory.")
        raise
    
    print("Keeping only necessary columns: Text and Score...")
    df = df[['Text', 'Score']]
    
    # Convert scores to binary sentiment (1-3: Negative, 4-5: Positive)
    # For binary classification, scores 4-5 are positive and 1-3 are negative
    # As per group decision, score 3 is treated as negative
    print("Converting scores to binary sentiment...")
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x >= 4 else 'Negative')
    print("\nSentiment distribution after conversion:")
    print(df['Sentiment'].value_counts())
    
    # Basic data exploration
    print(f"Dataset shape: {df.shape}")
    print("\nClass distribution:")
    print(df['Score'].value_counts())
    
    # Use only a subset of the data for faster processing
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Feature engineering - using only the review text for simplicity
    print("\nPreprocessing data...")
    X = df_sample['Text']  # Using only the review text as features
    y = df_sample['Sentiment']  # Target variable is sentiment (Positive/Negative)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)  # Limiting features for simplicity
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


def train_model_v1(X_train, y_train, X_test, y_test):
    """Train and evaluate version 1 of the model (Logistic Regression)"""
    print("\n" + "=" * 50)
    print("VERSION 1: Training model with Logistic Regression")
    print("=" * 50)
    
    # Display class distribution in training set
    print("\nTraining set class distribution:")
    print(pd.Series(y_train).value_counts())
    
    # Create a Logistic Regression model for binary sentiment classification
    print("\nCreating and training Logistic Regression model for sentiment analysis...")
    lr = LogisticRegression(
        C=1.0,                  # Inverse of regularization strength
        max_iter=100,           # Maximum number of iterations
        solver='liblinear',     # Algorithm to use
        random_state=42,
        verbose=1
    )
    
    # Train the model
    lr.fit(X_train, y_train)
    
    # Make predictions
    print("\nEvaluating model...")
    y_pred = lr.predict(X_test)
    
    # Evaluate the model
    evaluate_model(y_test, y_pred, "v1_logistic_regression")
    
    return lr


def train_model_v2(X_train, y_train, X_test, y_test):
    """Train and evaluate version 2 of the model (Random Forest)"""
    print("\n" + "=" * 50)
    print("VERSION 2: Training model with Random Forest")
    print("=" * 50)
    
    # Display class distribution in training set
    print("\nTraining set class distribution:")
    print(pd.Series(y_train).value_counts())
    
    # Create a Random Forest model for binary sentiment classification
    print("\nCreating and training Random Forest model for sentiment analysis...")
    rf = RandomForestClassifier(
        n_estimators=100,        # Number of trees
        max_depth=None,          # Maximum depth of trees
        min_samples_split=2,     # Minimum samples required to split
        random_state=42,
        verbose=1,
        n_jobs=-1                # Use all available cores
    )
    
    # Train the model
    rf.fit(X_train, y_train)
    
    # Make predictions
    print("\nEvaluating model...")
    y_pred = rf.predict(X_test)
    
    # Evaluate the model
    evaluate_model(y_test, y_pred, "v2_random_forest")
    
    return rf


def train_model_v3(X_train, y_train, X_test, y_test):
    """Train and evaluate version 3 of the model (Linear Support Vector Classifier)"""
    print("\n" + "=" * 50)
    print("VERSION 3: Training model with Linear Support Vector Classifier")
    print("=" * 50)
    
    # Display class distribution in training set
    print("\nTraining set class distribution:")
    print(pd.Series(y_train).value_counts())
    
    # Create a LinearSVC model for binary sentiment classification
    # This is much faster than SVC for large datasets
    print("\nCreating and training LinearSVC model for sentiment analysis...")
    linear_svc = LinearSVC(
        C=1.0,                  # Regularization parameter
        loss='hinge',           # Loss function
        max_iter=1000,          # Maximum number of iterations
        dual=True,              # Dual or primal formulation
        random_state=42,
        verbose=1
    )
    
    # Wrap the LinearSVC in a CalibratedClassifierCV to get probability estimates
    # This is needed because LinearSVC doesn't support probability=True
    print("\nCalibrating classifier to enable probability estimates...")
    svc = CalibratedClassifierCV(linear_svc, cv=3)
    
    # Train the model
    print("\nTraining model...")
    svc.fit(X_train, y_train)
    
    # Make predictions
    print("\nEvaluating model...")
    y_pred = svc.predict(X_test)
    
    # Evaluate the model
    evaluate_model(y_test, y_pred, "v3_linear_svc")
    
    return svc


def train_model_v4(X_train, y_train, X_test, y_test):
    """Train and evaluate version 1 of the model (imbalanced dataset)"""
    print("\n" + "=" * 50)
    print("VERSION 4: Training model with imbalanced dataset")
    print("=" * 50)
    
    # Display class distribution in training set
    print("\nTraining set class distribution:")
    print(pd.Series(y_train).value_counts())
    
    # Create a simple MLP model for binary sentiment classification
    print("\nCreating and training MLP model for sentiment analysis...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),  # One hidden layer with 100 neurons
        max_iter=100,               # Maximum number of iterations
        alpha=0.0001,               # L2 penalty parameter
        solver='adam',              # Optimizer
        random_state=42,
        verbose=True
    )
    
    # Train the model
    mlp.fit(X_train, y_train)
    
    # Make predictions
    print("\nEvaluating model...")
    y_pred = mlp.predict(X_test)
    
    # Evaluate the model
    evaluate_model(y_test, y_pred, "v4_imbalanced")
    
    return mlp


def train_model_v5(X_train, y_train, X_test, y_test):
    """Train and evaluate version 2 of the model (balanced dataset)"""
    print("\n" + "=" * 50)
    print("VERSION 5: Training model with balanced dataset")
    print("=" * 50)
    
    # Display original class distribution
    print("\nOriginal training set class distribution:")
    print(pd.Series(y_train).value_counts())
    
    # Balance the dataset using random undersampling
    print("\nBalancing the dataset...")
    rus = RandomUnderSampler(random_state=42)
    X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
    
    # Display balanced class distribution
    print("\nBalanced training set class distribution:")
    print(pd.Series(y_train_balanced).value_counts())
    
    # Create a simple MLP model for binary sentiment classification
    print("\nCreating and training MLP model with balanced data...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),  # One hidden layer with 100 neurons
        max_iter=100,               # Maximum number of iterations
        alpha=0.0001,               # L2 penalty parameter
        solver='adam',              # Optimizer
        random_state=42,
        verbose=True
    )
    
    # Train the model with balanced data
    mlp.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    print("\nEvaluating model...")
    y_pred = mlp.predict(X_test)
    
    # Evaluate the model
    evaluate_model(y_test, y_pred, "v5_balanced")
    
    return mlp


# Dictionary to store model results for comparison
model_results = {}

def evaluate_model(y_true, y_pred, model_version):
    """Evaluate model performance and create visualizations"""
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Positive')
    recall = recall_score(y_true, y_pred, pos_label='Positive')
    f1 = f1_score(y_true, y_pred, pos_label='Positive')
    
    # Store results for later comparison
    model_results[model_version] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('True Sentiment')
    plt.title(f'Sentiment Analysis Confusion Matrix - {model_version}')
    
    # Save to output directory
    output_dir = os.path.join(get_project_root(), 'output', 'visualizations')
    ensure_dir_exists(output_dir)
    output_path = os.path.join(output_dir, f'sentiment_confusion_matrix_{model_version}.png')
    plt.savefig(output_path)
    print(f"\nConfusion matrix saved as '{output_path}'")
    
    return accuracy


def generate_model_comparison_table():
    """Generate a comparison table of all model results"""
    if not model_results:
        print("No model results available for comparison")
        return
    
    # Prepare data for the table
    data = []
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    
    # Define the order of models to display
    model_order = [
        "v1_logistic_regression", 
        "v2_random_forest", 
        "v3_linear_svc",
        "v4_imbalanced", 
        "v5_balanced", 
        "v6_advanced"
    ]
    
    # Create friendly names for the models
    model_names = {
        "v1_logistic_regression": "Logistic Regression",
        "v2_random_forest": "Random Forest",
        "v3_svc": "Support Vector Classifier",
        "v3_linear_svc": "Linear SVC",
        "v4_imbalanced": "MLP (Imbalanced)",
        "v5_balanced": "MLP (Balanced)",
        "v6_advanced": "Deep Learning"
    }
    
    # Add data for each model in the specified order
    for model_id in model_order:
        if model_id in model_results:
            result = model_results[model_id]
            data.append([
                model_names.get(model_id, model_id),
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1']:.4f}"
            ])
    
    # Generate the table
    table = tabulate(data, headers=headers, tablefmt="grid")
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(table)
    
    # Save the table to a file
    output_dir = os.path.join(get_project_root(), 'output', 'results')
    ensure_dir_exists(output_dir)
    output_path = os.path.join(output_dir, 'model_comparison_results.txt')
    
    with open(output_path, 'w') as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(table)
    
    print(f"\nComparison table saved to '{output_path}'")
    
    # Create a bar chart comparing the metrics
    create_comparison_chart()


def create_comparison_chart():
    """Create a bar chart comparing model performance metrics"""
    if not model_results:
        return
    
    # Prepare data for plotting
    model_names = {
        "v1_logistic_regression": "Logistic Regression",
        "v2_random_forest": "Random Forest",
        "v3_svc": "Support Vector Classifier",
        "v3_linear_svc": "Linear SVC",
        "v4_imbalanced": "MLP (Imbalanced)",
        "v5_balanced": "MLP (Balanced)",
        "v6_advanced": "Deep Learning"
    }
    
    model_order = [
        "v1_logistic_regression", 
        "v2_random_forest", 
        "v3_linear_svc",
        "v4_imbalanced", 
        "v5_balanced", 
        "v6_advanced"
    ]
    
    # Filter to only include models that have results
    available_models = [m for m in model_order if m in model_results]
    
    # Prepare data
    models = [model_names.get(m, m) for m in available_models]
    accuracy = [model_results[m]['accuracy'] for m in available_models]
    precision = [model_results[m]['precision'] for m in available_models]
    recall = [model_results[m]['recall'] for m in available_models]
    f1 = [model_results[m]['f1'] for m in available_models]
    
    # Set up the plot
    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars
    
    fig, ax = plt.figure(figsize=(12, 8)), plt.axes()
    
    # Plot bars
    rects1 = ax.bar(x - width*1.5, accuracy, width, label='Accuracy')
    rects2 = ax.bar(x - width/2, precision, width, label='Precision')
    rects3 = ax.bar(x + width/2, recall, width, label='Recall')
    rects4 = ax.bar(x + width*1.5, f1, width, label='F1 Score')
    
    # Add labels and title
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    fig.tight_layout()
    
    # Save the chart
    output_dir = os.path.join(get_project_root(), 'output', 'visualizations')
    ensure_dir_exists(output_dir)
    output_path = os.path.join(output_dir, 'model_comparison_chart.png')
    plt.savefig(output_path)
    print(f"\nComparison chart saved as '{output_path}'")
    plt.close()


# Load and preprocess data
X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = load_and_preprocess_data(sample_size=50000)

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for a given text"""
    # Vectorize the text
    text_tfidf = vectorizer.transform([text])
    # Predict sentiment
    prediction = model.predict(text_tfidf)[0]
    return prediction


def test_example_predictions(model, vectorizer, model_version):
    """Test the model on example texts"""
    print(f"\nExample sentiment predictions for {model_version}:")
    example_texts = [
        "This product is amazing! I love it and would definitely recommend it to others.",
        "Terrible experience. The product broke after one use and customer service was unhelpful."
    ]

    for text in example_texts:
        sentiment = predict_sentiment(text, model, vectorizer)
        print(f"Text: {text[:50]}...")
        print(f"Predicted sentiment: {sentiment}\n")


def train_model_v6(X_train, y_train, X_test, y_test):
    """Train and evaluate version 3 of the model with advanced techniques
    
    This version includes:
    - L2 regularization
    - Dropout layers
    - Batch normalization
    - Early stopping
    - Multiple hidden layers
    - Balanced dataset
    """
    print("\n" + "=" * 50)
    print("VERSION 6: Training model with advanced techniques")
    print("=" * 50)
    
    # Display original class distribution
    print("\nOriginal training set class distribution:")
    print(pd.Series(y_train).value_counts())
    
    # Balance the dataset using random undersampling
    print("\nBalancing the dataset...")
    rus = RandomUnderSampler(random_state=42)
    X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
    
    # Display balanced class distribution
    print("\nBalanced training set class distribution:")
    print(pd.Series(y_train_balanced).value_counts())
    
    # Convert labels to numerical format
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_balanced)
    y_test_encoded = le.transform(y_test)
    
    # Convert to dense numpy arrays if they are sparse matrices
    if hasattr(X_train_balanced, 'toarray'):
        X_train_dense = X_train_balanced.toarray()
    else:
        X_train_dense = X_train_balanced
        
    if hasattr(X_test, 'toarray'):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test
    
    # Create an advanced neural network model
    print("\nCreating advanced neural network model...")
    model = Sequential([
        # Input layer with L2 regularization
        Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train_dense.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),  # 50% dropout to prevent overfitting
        
        # Hidden layer 1
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Hidden layer 2
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Set up callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Set up model directory
    models_dir = os.path.join(get_project_root(), 'models', 'saved')
    ensure_dir_exists(models_dir)
    model_path = os.path.join(models_dir, 'best_model_v6.h5')
    
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    print("\nTraining advanced model...")
    history = model.fit(
        X_train_dense, y_train_encoded,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    # Save to output directory
    output_dir = os.path.join(get_project_root(), 'output', 'visualizations')
    ensure_dir_exists(output_dir)
    output_path = os.path.join(output_dir, 'v3_training_history.png')
    # Update the filename to reflect the new version number
    output_path = os.path.join(output_dir, 'v6_training_history.png')
    plt.savefig(output_path)
    print(f"\nTraining history plot saved as '{output_path}'")
    
    # Evaluate the model
    print("\nEvaluating model...")
    y_pred_prob = model.predict(X_test_dense)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_pred = le.inverse_transform(y_pred.flatten())
    
    # Evaluate the model
    evaluate_model(y_test, y_pred, "v6_advanced")
    
    return model, le


def predict_sentiment_v6(text, model, vectorizer, label_encoder):
    """Predict sentiment for a given text using the advanced model"""
    # Vectorize the text
    text_tfidf = vectorizer.transform([text])
    
    # Convert to dense array if it's a sparse matrix
    if hasattr(text_tfidf, 'toarray'):
        text_tfidf = text_tfidf.toarray()
    
    # Predict sentiment
    prediction_prob = model.predict(text_tfidf)
    prediction = (prediction_prob > 0.5).astype(int)
    sentiment = label_encoder.inverse_transform(prediction.flatten())[0]
    
    return sentiment


def test_example_predictions_v6(model, vectorizer, label_encoder, model_version):
    """Test the advanced model on example texts"""
    print(f"\nExample sentiment predictions for {model_version}:")
    example_texts = [
        "This product is amazing! I love it and would definitely recommend it to others.",
        "Terrible experience. The product broke after one use and customer service was unhelpful."
    ]

    for text in example_texts:
        sentiment = predict_sentiment_v6(text, model, vectorizer, label_encoder)
        print(f"Text: {text[:50]}...")
        print(f"Predicted sentiment: {sentiment}\n")


def save_model_for_integration(model, vectorizer, model_name="best_model"):
    """Save the model, vectorizer, and label encoder for integration"""
    import joblib
    from sklearn.preprocessing import LabelEncoder
    
    # Create a label encoder for converting between 'Positive'/'Negative' and 0/1
    label_encoder = LabelEncoder()
    label_encoder.fit(['Negative', 'Positive'])
    
    # Save the model
    print(f"\nSaving {model_name} for integration...")
    models_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save the model
    joblib.dump(model, os.path.join(models_dir, 'sentiment_model.pkl'))
    print(f"Model saved as 'sentiment_model.pkl'")
    
    # Save the vectorizer
    joblib.dump(vectorizer, os.path.join(models_dir, 'vectorizer.pkl'))
    print(f"Vectorizer saved as 'vectorizer.pkl'")
    
    # Save the label encoder
    joblib.dump(label_encoder, os.path.join(models_dir, 'label_encoder.pkl'))
    print(f"Label encoder saved as 'label_encoder.pkl'")
    
    print(f"\nAll files saved successfully in {models_dir}")
    print("The model is now ready for integration with the sentiment analyzer.")


def main():
    """Main function to run all model versions"""
    # Load and preprocess data
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = load_and_preprocess_data(sample_size=50000)
    
    # Train and evaluate version 1 (Logistic Regression)
    model_v1 = train_model_v1(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Test example predictions with version 1
    test_example_predictions(model_v1, vectorizer, "Version 1 (Logistic Regression)")
    
    # Train and evaluate version 2 (Random Forest)
    model_v2 = train_model_v2(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Test example predictions with version 2
    test_example_predictions(model_v2, vectorizer, "Version 2 (Random Forest)")
    
    # Train and evaluate version 3 (SVC)
    model_v3 = train_model_v3(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Test example predictions with version 3
    test_example_predictions(model_v3, vectorizer, "Version 3 (SVC)")
    
    # Train and evaluate version 4 (imbalanced dataset)
    model_v4 = train_model_v4(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Test example predictions with version 4
    test_example_predictions(model_v4, vectorizer, "Version 4 (MLP with imbalanced dataset)")
    
    # Train and evaluate version 5 (balanced dataset)
    model_v5 = train_model_v5(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Test example predictions with version 5
    test_example_predictions(model_v5, vectorizer, "Version 5 (MLP with balanced dataset)")
    
    # Train and evaluate version 6 (advanced model)
    model_v6, label_encoder = train_model_v6(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Test example predictions with version 6
    test_example_predictions_v6(model_v6, vectorizer, label_encoder, "Version 6 (Advanced)")
    
    # Get the model path for the final message
    models_dir = os.path.join(get_project_root(), 'models', 'saved')
    model_path = os.path.join(models_dir, 'best_model_v6.h5')
    
    # Generate comparison table and visualization
    generate_model_comparison_table()
    
    # Save the best model (MLP Imbalanced - model_v4) for integration
    # Based on the comparison results, this model has the best F1 score
    save_model_for_integration(model_v4, vectorizer, "MLP Imbalanced (v4)")
    
    print("\nModel comparison complete. Check the confusion matrices and classification reports for detailed results.")
    print(f"\nAdvanced model (v6) has been saved as '{model_path}'.")


# Run the main function
if __name__ == "__main__":
    main()
