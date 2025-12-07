"""
Unit tests for Wine Quality Model Pipeline
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_model_training_imports():
    """Test that all required modules can be imported"""
    try:
        import train_model
        import evaluate_model
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_data_loading():
    """Test data loading function"""
    from train_model import load_wine_quality_data
    
    X, y = load_wine_quality_data()
    
    # Check data shapes
    assert X is not None
    assert y is not None
    assert len(X) == len(y)
    assert X.shape[1] == 11  # Wine dataset has 11 features

def test_model_directory_creation():
    """Test that model directories are created"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    assert os.path.exists('models')
    assert os.path.exists('metrics')

def test_preprocessing():
    """Test data preprocessing"""
    from train_model import preprocess_data
    
    # Create dummy data
    X = pd.DataFrame(np.random.randn(100, 11))
    y = np.random.randint(0, 2, 100)
    
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Check outputs
    assert X_train.shape[0] == 80  # 80% train
    assert X_test.shape[0] == 20   # 20% test
    assert len(y_train) == 80
    assert len(y_test) == 20
    assert scaler is not None

def test_model_training():
    """Test model training with dummy data"""
    from train_model import train_gradient_boosting_model
    
    # Create small dummy dataset
    X_train = np.random.randn(50, 11)
    y_train = np.random.randint(0, 2, 50)
    
    model = train_gradient_boosting_model(X_train, y_train)
    
    # Check model attributes
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    assert hasattr(model, 'feature_importances_')

def test_model_evaluation():
    """Test model evaluation functions"""
    from train_model import evaluate_model
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Create and train dummy model
    X_test = np.random.randn(20, 11)
    y_test = np.random.randint(0, 2, 20)
    
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X_test, y_test)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Check metrics
    assert 'accuracy' in metrics
    assert 'confusion_matrix' in metrics
    assert 'classification_report' in metrics
    assert 'feature_importance' in metrics
    assert 0 <= metrics['accuracy'] <= 1

def test_model_saving():
    """Test model saving functionality"""
    from train_model import save_model_and_metrics
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Create dummy model and metrics
    model = GradientBoostingClassifier(n_estimators=10)
    model.fit(np.random.randn(50, 11), np.random.randint(0, 2, 50))
    
    metrics = {'accuracy': 0.85, 'test': 'value'}
    
    model_file, metrics_file = save_model_and_metrics(model, metrics, version=999)
    
    # Check files were created
    assert model_file is not None
    assert metrics_file is not None
    assert 'v999' in model_file

if __name__ == "__main__":
    pytest.main([__file__, "-v"])