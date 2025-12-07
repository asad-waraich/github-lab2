"""
Custom Training Script for Wine Quality Prediction
Uses Gradient Boosting Classifier instead of Random Forest
Dataset: Wine Quality Dataset
Author: Your Name
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_wine_quality_data():
    """
    Load Wine Quality dataset
    Using red wine dataset from UCI repository
    """
    print("Loading Wine Quality dataset...")
    
    # Wine Quality dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    try:
        # Load data
        data = pd.read_csv(url, sep=';')
        
        # Basic data info
        print(f"Dataset shape: {data.shape}")
        print(f"Features: {list(data.columns[:-1])}")
        
        # Prepare features and target
        X = data.drop('quality', axis=1)
        
        # Convert to binary classification: good (quality >= 7) vs not good
        y = (data['quality'] >= 7).astype(int)
        
        print(f"Class distribution - Good wines (quality >= 7): {y.sum()} / {len(y)}")
        
        return X, y
    
    except Exception as e:
        print(f"Error loading data, using synthetic data instead: {e}")
        # Fallback to synthetic data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=11, n_informative=8, 
                                  n_redundant=2, n_clusters_per_class=1,
                                  weights=[0.7, 0.3], random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(11)])
        return X, y

def preprocess_data(X, y):
    """
    Preprocess data: split and scale
    """
    print("\nPreprocessing data...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_gradient_boosting_model(X_train, y_train):
    """
    Train a Gradient Boosting Classifier
    """
    print("\nTraining Gradient Boosting Classifier...")
    
    # Model with custom parameters
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation
    """
    print("\nEvaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    feature_importance = model.feature_importances_
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:\n{conf_matrix}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist(),
        'feature_importance': feature_importance.tolist()
    }

def save_model_and_metrics(model, metrics, version=None):
    """
    Save model and metrics with versioning
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Version tracking
    if version is None:
        version_file = 'models/version.txt'
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version = int(f.read().strip()) + 1
        else:
            version = 1
        
        with open(version_file, 'w') as f:
            f.write(str(version))
    
    # Save model
    model_filename = f'models/gradient_boost_model_v{version}_{timestamp}.pkl'
    joblib.dump(model, model_filename)
    print(f"\nModel saved as: {model_filename}")
    
    # Save metrics
    metrics['version'] = version
    metrics['timestamp'] = timestamp
    metrics['model_type'] = 'GradientBoostingClassifier'
    
    metrics_filename = f'metrics/metrics_v{version}_{timestamp}.json'
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved as: {metrics_filename}")
    
    # Save latest model link
    latest_model = 'models/latest_model.pkl'
    joblib.dump(model, latest_model)
    
    return model_filename, metrics_filename

def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("Wine Quality Prediction - Custom ML Pipeline")
    print("="*60)
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Load data
    X, y = load_wine_quality_data()
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train model
    model = train_gradient_boosting_model(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save everything
    model_file, metrics_file = save_model_and_metrics(model, metrics)
    
    print("\n" + "="*60)
    print("Training Pipeline Completed Successfully!")
    print(f"Model: {model_file}")
    print(f"Metrics: {metrics_file}")
    print("="*60)
    
    return model, metrics

if __name__ == "__main__":
    main()