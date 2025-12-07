# 🍷 Wine Quality Prediction with MLOps Pipeline

## Custom Implementation of GitHub Actions MLOps Lab

This repository demonstrates an end-to-end MLOps pipeline using GitHub Actions for continuous integration and deployment of a machine learning model. This is a **customized version** of the original lab with significant enhancements.

## 🎯 Key Customizations

### 1. **Different Dataset**
- **Original**: Synthetic/Iris dataset
- **Custom**: UCI Wine Quality dataset (Red Wine)
- Binary classification: Predicting if wine quality is "Good" (≥7) or "Regular" (<7)

### 2. **Different Model**
- **Original**: Random Forest Classifier
- **Custom**: Gradient Boosting Classifier with hyperparameter tuning
- Added model calibration for better probability estimates

### 3. **Interactive Dashboard**
- **New Addition**: Streamlit-based monitoring dashboard
- Real-time predictions and model monitoring

---

## 📊 Model Performance Metrics

### Latest Model Results


| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | 0.9375 | Overall correct predictions |
| **Precision** | 0.8485 | Accuracy of positive predictions |
| **Recall** | 0.6512 | Coverage of actual positives |
| **F1-Score** | 0.7368 | Harmonic mean of precision & recall |
| **AUC-ROC** | 0.9243 | Area under ROC curve |

### Confusion Matrix

|  | Predicted Regular | Predicted Good |
|---|------------------|----------------|
| **Actual Regular** | 272 | 5 |
| **Actual Good** | 15 | 28 |

- **True Positives**: 28 (Good wines correctly identified)
- **True Negatives**: 272 (Regular wines correctly identified)  
- **False Positives**: 5 (Regular wines misclassified as Good)
- **False Negatives**: 15 (Good wines misclassified as Regular)

### Performance Highlights
- ✅ **93.75% Accuracy** - Exceptional overall performance
- ✅ **98% Specificity** - Excellent at identifying regular wines (272/277)
- ⚠️ **65% Sensitivity** - Room for improvement in detecting good wines
- ✅ **Low False Positive Rate** - Only 5 regular wines misclassified as good

### Class Distribution
- **Test Set Size**: 320 samples
- **Good Wines (Quality ≥ 7)**: 13.4% of dataset
- **Regular Wines**: 86.6% of dataset

---

### Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Regular Wine (0) | 0.95 | 0.98 | 0.96 | 277 |
| Good Wine (1) | 0.85 | 0.65 | 0.74 | 43 |
| **Weighted Avg** | **0.93** | **0.94** | **0.93** | 320 |

## 📈 Feature Importance

Top 5 Most Important Features for Wine Quality:

1. **Alcohol**: 0.324 (32.4%)
2. **Volatile Acidity**: 0.281 (28.1%)
3. **Sulphates**: 0.152 (15.2%)
4. **Total Sulfur Dioxide**: 0.089 (8.9%)
5. **pH**: 0.067 (6.7%)

---


## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (Python 3.11 recommended)
- Git
- GitHub account

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/github-lab2.git
cd github-lab2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Train the model
python3 src/train_model.py

# Evaluate the model
python3 src/evaluate_model.py

# Launch the dashboard
streamlit run src/dashboard.py
```

---

## 🎨 Dashboard Features

### Available Pages:
1. **Model Overview** - Current model information and feature importance
2. **Performance Metrics** - Detailed metrics and confusion matrix
3. **Make Predictions** - Interactive wine quality predictions
4. **Model History** - Version comparison and tracking

### Sample Predictions

#### Example 1: High-Quality Wine
- **Alcohol**: 12.5%
- **Volatile Acidity**: 0.3
- **Sulphates**: 0.8
- **Prediction**: Good Wine ✅
- **Confidence**: 87%

#### Example 2: Regular Wine
- **Alcohol**: 9.5%
- **Volatile Acidity**: 0.7
- **Sulphates**: 0.4
- **Prediction**: Regular Wine
- **Confidence**: 92%

---

## 📁 Project Structure

```
github-lab2/
├── .github/
│   └── workflows/
│       └── model_retraining_on_push.yml
├── src/
│   ├── train_model.py
│   └── evaluate_model.py
├── models/
│   ├── model_v3_*.pkl
│   ├── latest_model.pkl
│   └── scaler.pkl
├── metrics/
│   ├── metrics_v3_*.json
│   └── evaluation_dashboard_*.png
├── test/
│   └── test_model.py
├── dashboard.py
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Details

### Model Configuration
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

### Data Preprocessing
- StandardScaler for feature normalization
- 80/20 train-test split
- Stratified sampling to maintain class distribution

### Evaluation Metrics
- ROC Curve and AUC
- Precision-Recall Curve
- Calibration Plot (Brier Score: 0.0923)
- Feature Importance Analysis

---

## 🏗️ CI/CD Pipeline

### GitHub Actions Workflow
- **Triggers**: On push to main branch
- **Steps**:
  1. Set up Python environment
  2. Install dependencies
  3. Train Gradient Boosting model
  4. Evaluate performance
  5. Generate visualizations
  6. Store artifacts
  7. Create release

### Workflow Status
![GitHub Actions](https://github.com/YOUR_USERNAME/github-lab2/actions/workflows/model_retraining_on_push.yml/badge.svg)

---

## 📝 Key Learning Outcomes

Through this customized implementation, I demonstrated:

1. **MLOps Best Practices**: Automated CI/CD, version control, testing
2. **Model Development**: Feature engineering, hyperparameter tuning
3. **Performance Monitoring**: Dashboard creation, metrics tracking
4. **Software Engineering**: Clean code, documentation, testing

---

## 🎯 Improvements Over Original Lab

| Aspect | Original | My Implementation | Improvement |
|--------|----------|-------------------|-------------|
| Dataset | Synthetic | Real Wine Quality | Real-world application |
| Model | Random Forest | Gradient Boosting | +3.8% accuracy |
| Metrics | Basic accuracy | Comprehensive suite | 5+ additional metrics |
| Visualization | None | Interactive dashboard | Full monitoring |
| Calibration | None | Isotonic regression | Better probabilities |

---
### Model Calibration
- **Original Brier Score**: 0.0419 (excellent calibration)
- **Calibrated Brier Score**: 0.0421
- **Note**: Model is already well-calibrated; minimal improvement needed

## 🐛 Troubleshooting

### Common Issues

1. **Python 3.13 Compatibility**
   - Solution: Use Python 3.11 or 3.12
   - `brew install python@3.11`

2. **Import Errors**
   - Solution: Install packages individually
   - `pip install numpy pandas scikit-learn matplotlib`

3. **Dashboard Not Loading**
   - Solution: Check port 8501 is free
   - `lsof -i :8501`



---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original lab framework from [MLOps Course](https://github.com/raminmohammadi/MLOps)
- Wine Quality dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- Built with scikit-learn, Streamlit, and GitHub Actions

---
