"""
Interactive Model Monitoring Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import glob

# Page configuration
st.set_page_config(
    page_title="Wine Quality ML Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model_versions():
    """Load all available model versions"""
    model_files = glob.glob('models/gradient_boost_model_v*.pkl')
    versions = []
    for file in model_files:
        # Extract version and timestamp from filename
        parts = file.split('_')
        version = parts[3].replace('v', '')
        timestamp = parts[4].replace('.pkl', '')
        versions.append({
            'version': int(version),
            'timestamp': timestamp,
            'file': file
        })
    return sorted(versions, key=lambda x: x['version'], reverse=True)

def load_metrics_history():
    """Load metrics history for all versions"""
    metrics_files = glob.glob('metrics/metrics_v*.json')
    history = []
    
    for file in metrics_files:
        with open(file, 'r') as f:
            data = json.load(f)
            history.append(data)
    
    return sorted(history, key=lambda x: x.get('version', 0))

def display_model_info(model, version_info):
    """Display model information"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "Gradient Boosting")
        st.metric("Version", f"v{version_info['version']}")
    
    with col2:
        st.metric("Training Date", version_info['timestamp'][:8])
        st.metric("Training Time", version_info['timestamp'][9:])
    
    with col3:
        if hasattr(model, 'n_estimators'):
            st.metric("Trees", model.n_estimators)
        if hasattr(model, 'max_depth'):
            st.metric("Max Depth", model.max_depth)

def plot_metrics_evolution(history):
    """Plot metrics evolution over versions"""
    if not history:
        st.warning("No metrics history available")
        return
    
    versions = [h['version'] for h in history]
    accuracies = [h.get('accuracy', 0) for h in history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=versions,
        y=accuracies,
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#8B0000', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Model Performance Evolution",
        xaxis_title="Version",
        yaxis_title="Accuracy",
        hovermode='x',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def predict_wine_quality(model, scaler):
    """Interactive prediction interface"""
    st.subheader("🔮 Predict Wine Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 2.0, 0.5)
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
        residual_sugar = st.slider("Residual Sugar", 0.5, 16.0, 2.5)
        chlorides = st.slider("Chlorides", 0.01, 0.61, 0.08)
        free_sulfur_dioxide = st.slider("Free SO2", 1.0, 72.0, 15.0)
    
    with col2:
        total_sulfur_dioxide = st.slider("Total SO2", 6.0, 290.0, 45.0)
        density = st.slider("Density", 0.99, 1.01, 0.996)
        pH = st.slider("pH", 2.7, 4.0, 3.3)
        sulphates = st.slider("Sulphates", 0.3, 2.0, 0.65)
        alcohol = st.slider("Alcohol", 8.0, 15.0, 10.5)
    
    if st.button("Predict Quality", type="primary"):
        # Prepare input
        input_data = np.array([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol
        ]])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            quality = "🍷 Good Wine" if prediction == 1 else "Regular Wine"
            st.success(f"**Prediction:** {quality}")
        
        with col2:
            confidence = max(probability) * 100
            st.info(f"**Confidence:** {confidence:.1f}%")
        
        # Probability bar chart
        fig = go.Figure(data=[
            go.Bar(x=['Regular', 'Good'], y=probability,
                   marker_color=['#FFA07A', '#8B0000'])
        ])
        fig.update_layout(
            title="Quality Probability Distribution",
            yaxis_title="Probability",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🍷 Wine Quality ML Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Model Overview", "Performance Metrics", "Make Predictions", "Model History"]
    )
    
    # Load data
    versions = load_model_versions()
    
    if not versions:
        st.error("No trained models found. Please run train_model.py first.")
        return
    
    # Model selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Selection")
    selected_version = st.sidebar.selectbox(
        "Select Version",
        [v['version'] for v in versions],
        format_func=lambda x: f"Version {x}"
    )
    
    # Load selected model
    version_info = next(v for v in versions if v['version'] == selected_version)
    model = joblib.load(version_info['file'])
    scaler = joblib.load('models/scaler.pkl') if os.path.exists('models/scaler.pkl') else None
    
    # Page routing
    if page == "Model Overview":
        st.header("📊 Model Overview")
        display_model_info(model, version_info)
        
        st.markdown("---")
        st.subheader("About This Model")
        st.info("""
        This **Gradient Boosting Classifier** is trained on the Wine Quality dataset 
        to predict whether a wine is of good quality (rating ≥ 7) based on its 
        chemical properties.
        
        The model uses 11 features including acidity levels, sugar content, and alcohol 
        percentage to make predictions.
        """)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 
                           'residual sugar', 'chlorides', 'free sulfur dioxide',
                           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h', color='Importance',
                        color_continuous_scale='Reds')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Performance Metrics":
        st.header("📈 Performance Metrics")
        
        # Load latest metrics
        metrics_files = glob.glob(f'metrics/metrics_v{selected_version}_*.json')
        if metrics_files:
            with open(metrics_files[0], 'r') as f:
                metrics = json.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                report = metrics.get('classification_report', {})
                if '1' in report:
                    st.metric("Precision", f"{report['1']['precision']:.3f}")
            with col3:
                if '1' in report:
                    st.metric("Recall", f"{report['1']['recall']:.3f}")
            with col4:
                if '1' in report:
                    st.metric("F1-Score", f"{report['1']['f1-score']:.3f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual"),
                              x=['Regular', 'Good'], y=['Regular', 'Good'],
                              color_continuous_scale='Blues', text_auto=True)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Metrics evolution
        st.subheader("Performance Over Time")
        history = load_metrics_history()
        plot_metrics_evolution(history)
    
    elif page == "Make Predictions":
        if scaler is not None:
            predict_wine_quality(model, scaler)
        else:
            st.error("Scaler not found. Please retrain the model.")
    
    elif page == "Model History":
        st.header("📜 Model History")
        
        # Display all versions
        st.subheader("Available Versions")
        for v in versions:
            with st.expander(f"Version {v['version']} - {v['timestamp']}"):
                st.write(f"**File:** {v['file']}")
                
                # Load metrics for this version
                metrics_files = glob.glob(f"metrics/metrics_v{v['version']}_*.json")
                if metrics_files:
                    with open(metrics_files[0], 'r') as f:
                        metrics = json.load(f)
                    st.json(metrics)
        
        # Comparison table
        st.subheader("Version Comparison")
        history = load_metrics_history()
        if history:
            comparison_df = pd.DataFrame([
                {
                    'Version': h['version'],
                    'Accuracy': h.get('accuracy', 0),
                    'Timestamp': h.get('timestamp', 'N/A')
                }
                for h in history
            ])
            st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main()