"""
Enhanced Model Evaluation Script with Visualizations
Fixed imports for compatibility
"""

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import os
from datetime import datetime

def load_latest_model():
    """Load the most recent model"""
    model_path = 'models/latest_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # Try to find any model
        import glob
        model_files = glob.glob('models/model_v*.pkl')
        if model_files:
            return joblib.load(model_files[0])
        raise FileNotFoundError("No model found. Please run train_model.py first.")

def load_test_data():
    """Load and prepare test data"""
    # For evaluation, we'll regenerate the same test split
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    try:
        data = pd.read_csv(url, sep=';')
        X = data.drop('quality', axis=1)
        y = (data['quality'] >= 7).astype(int)
        
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Load and apply scaler if it exists
        if os.path.exists('models/scaler.pkl'):
            scaler = joblib.load('models/scaler.pkl')
            X_test_scaled = scaler.transform(X_test)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)
        
        return X_test_scaled, y_test, X.columns.tolist()
    
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic data for evaluation...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200, n_features=11, random_state=42)
        return X, y, [f'feature_{i}' for i in range(11)]

def create_evaluation_plots(model, X_test, y_test, feature_names):
    """Create comprehensive evaluation visualizations"""
    
    print("Creating evaluation plots...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. ROC Curve
    ax1 = plt.subplot(2, 3, 1)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    ax2 = plt.subplot(2, 3, 2)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax2.plot(recall, precision, color='green', lw=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(recall, precision, alpha=0.2, color='green')
    
    # 3. Feature Importance (if available)
    ax3 = plt.subplot(2, 3, 3)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10
        
        ax3.barh(range(len(indices)), importances[indices], color='skyblue')
        ax3.set_yticks(range(len(indices)))
        ax3.set_yticklabels([feature_names[i] if i < len(feature_names) else f'feature_{i}' 
                             for i in indices])
        ax3.set_xlabel('Importance')
        ax3.set_title('Top Feature Importances')
    else:
        ax3.text(0.5, 0.5, 'Feature importance\nnot available', 
                ha='center', va='center')
        ax3.set_title('Feature Importances')
    
    # 4. Calibration Plot
    ax4 = plt.subplot(2, 3, 4)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_pred_proba, n_bins=10
    )
    ax4.plot(mean_predicted_value, fraction_of_positives, 
            marker='o', linewidth=1, label='Model')
    ax4.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    ax4.set_xlabel('Mean Predicted Probability')
    ax4.set_ylabel('Fraction of Positives')
    ax4.set_title('Calibration Plot')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Confusion Matrix
    ax5 = plt.subplot(2, 3, 5)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Actual')
    ax5.set_title('Confusion Matrix')
    
    # 6. Score Distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.5, 
            label='Regular Wine', color='red', density=True)
    ax6.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.5, 
            label='Good Wine', color='green', density=True)
    ax6.set_xlabel('Predicted Probability')
    ax6.set_ylabel('Density')
    ax6.set_title('Score Distribution by Class')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('metrics', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'metrics/evaluation_dashboard_{timestamp}.png'
    plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
    print(f"Evaluation dashboard saved: {plot_filename}")
    

    
    return plot_filename

def calibrate_model(model, X_test, y_test):
    """Apply probability calibration"""
    print("\nCalibrating model probabilities...")
    
    # Check if we have enough samples
    if len(y_test) < 20:
        print("Not enough samples for calibration. Skipping...")
        return model, None
    
    try:
        # Split test set for calibration
        from sklearn.model_selection import train_test_split
        X_cal, X_val, y_cal, y_val = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )
        
        # Calibrate using isotonic regression
        calibrated_clf = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated_clf.fit(X_cal, y_cal)
        
        # Compare scores
        from sklearn.metrics import brier_score_loss
        original_proba = model.predict_proba(X_val)[:, 1]
        calibrated_proba = calibrated_clf.predict_proba(X_val)[:, 1]
        
        original_brier = brier_score_loss(y_val, original_proba)
        calibrated_brier = brier_score_loss(y_val, calibrated_proba)
        
        print(f"Original Brier Score: {original_brier:.4f}")
        print(f"Calibrated Brier Score: {calibrated_brier:.4f}")
        
        improvement = ((original_brier - calibrated_brier) / original_brier * 100)
        print(f"Improvement: {improvement:.2f}%")
        
        # Save calibrated model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibrated_model_path = f'models/calibrated_model_{timestamp}.pkl'
        joblib.dump(calibrated_clf, calibrated_model_path)
        print(f"Calibrated model saved: {calibrated_model_path}")
        
        return calibrated_clf, calibrated_brier
    
    except Exception as e:
        print(f"Calibration failed: {e}")
        return model, None

def generate_detailed_metrics(model, X_test, y_test):
    """Generate comprehensive metrics"""
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float((y_pred == y_test).mean()),
        'f1_score': float(f1_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'samples_evaluated': len(y_test),
        'positive_rate': float(y_test.mean()),
        'model_type': type(model).__name__
    }
    
    # Add AUC if we have probabilities
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['auc_roc'] = float(auc(fpr, tpr))
    except:
        metrics['auc_roc'] = None
    
    # Add classification report
    metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    
    # Add confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics

def main():
    """Main evaluation pipeline"""
    print("="*60)
    print("Enhanced Model Evaluation Pipeline")
    print("="*60)
    
    try:
        # Load model and data
        print("\nLoading model and data...")
        model = load_latest_model()
        X_test, y_test, feature_names = load_test_data()
        
        print(f"Model type: {type(model).__name__}")
        print(f"Test set size: {len(y_test)} samples")
        print(f"Positive class rate: {y_test.mean():.2%}")
        
        # Generate detailed metrics
        print("\nGenerating metrics...")
        metrics = generate_detailed_metrics(model, X_test, y_test)
        
        print("\nModel Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics['auc_roc']:
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        # Create visualizations
        plot_file = create_evaluation_plots(model, X_test, y_test, feature_names)
        
        # Calibrate model (optional)
        calibrated_model, brier_score = calibrate_model(model, X_test, y_test)
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'metrics': metrics,
            'calibration': {
                'calibrated': brier_score is not None,
                'brier_score': float(brier_score) if brier_score else None
            },
            'visualizations': plot_file,
            'timestamp': timestamp
        }
        
        results_file = f'metrics/evaluation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved: {results_file}")
        print("="*60)
        print("Evaluation Complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("Please make sure you've run train_model.py first.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()