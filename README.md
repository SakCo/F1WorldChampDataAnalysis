# 🏎️ Formula 1 World Championship Data Analysis & ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-red.svg)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue.svg)](https://matplotlib.org/)

## 🏁 Project Overview

**End-to-end data science project** analyzing Formula 1 World Championship data (1950-2020) using machine learning, statistical analysis, and interactive visualizations. Demonstrates complete ML lifecycle from data preprocessing to model optimization achieving **100% accuracy**.

### 🎯 Objective
Predict driver performance and race outcomes using historical F1 data while extracting actionable insights about racing patterns and performance factors.

## ✨ Key Features & Technical Skills

### 🔍 **Data Engineering**
- **Multi-source Integration**: Merged 6 datasets (25K+ records)
- **Feature Engineering**: Driver age calculation, performance metrics
- **Data Quality**: IQR-based outlier detection, missing value imputation
- **Statistical Analysis**: Skewness testing, correlation analysis

### 🗺️ **Geospatial Analytics**
- **Interactive Mapping**: Folium world map with F1 circuit locations
- **GPS Visualization**: Custom markers with circuit details

### 🤖 **Machine Learning Pipeline**
- **6 Classification Algorithms**: Logistic Regression, Random Forest, Decision Tree, KNN, Naive Bayes, SGD
- **3 Scaling Techniques**: MinMaxScaler, StandardScaler, RobustScaler
- **Performance Optimization**: Achieved 40%+ accuracy improvement through scaling

## 🗂️ Dataset & Performance

### 📋 **Data Sources** (1950-2020)

| Dataset | Records | Features | Description |
|---------|---------|----------|-------------|
| `results.csv` | 25,840+ | 18 | Race results and performance metrics |
| `drivers.csv` | 857 | 8 | Driver biographical information |
| `circuits.csv` | 77 | 9 | Circuit specifications and coordinates |

### 🎯 **Target Variables**

- **Driver Performance Classification**: Based on race finishing positions
- **Performance Metrics**: Points, lap times, fastest lap speeds

## 🤖 Machine Learning Models

### 🔬 **Algorithms Implemented**

1. **Logistic Regression** - Linear classification baseline
2. **Decision Tree Classifier** - Interpretable tree-based model with visualization
3. **Random Forest Classifier** - Ensemble method for improved accuracy
4. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
5. **Gaussian Naive Bayes** - Probabilistic classifier
6. **Stochastic Gradient Descent (SGD)** - Scalable optimization algorithm

### ⚙️ **Feature Scaling Techniques**

- **MinMaxScaler**: Range [0,1] normalization
- **StandardScaler**: Z-score standardization (μ=0, σ=1)
- **RobustScaler**: Median and IQR-based scaling (outlier-resistant)

### 📊 **Model Performance**

| Model | Raw Data | MinMaxScaler | StandardScaler | RobustScaler |
|-------|----------|--------------|----------------|--------------|
| Random Forest | 95%+ | 99%+ | 100% | 100% |
| Decision Tree | 90%+ | 95%+ | 100% | 100% |
| Logistic Regression | 60% | 99%+ | 100% | 99%+ |

## 📊 Visualization & Analysis

### 🎨 **Advanced Visualizations**

- **Interactive Correlation Heatmaps**: Feature relationship analysis
- **Geographic Circuit Mapping**: World map with F1 venue locations
- **Performance Trend Analysis**: Algorithm accuracy comparison charts
- **Decision Tree Visualization**: Complete model interpretability
- **Statistical Distribution Plots**: Data quality assessment visualizations

### 📈 **Key Insights Discovered**

- **Scaling Impact**: Demonstrated 40%+ accuracy improvement with proper feature scaling
- **Algorithm Performance**: Ensemble methods consistently outperformed single algorithms
- **Data Quality**: Outlier removal significantly improved model generalization
- **Feature Importance**: Identified critical performance predictors

## ⚡ Performance Results

**Key Achievements:**
- ✅ **100% Model Accuracy** achieved (Random Forest + StandardScaler)
- ✅ **67% Accuracy Improvement** through feature scaling optimization
- ✅ **6 ML Algorithms** successfully implemented and compared
- ✅ **Interactive Visualizations** with geospatial mapping

## 🛠️ Technologies Used

**Core Stack:** Python 3.8+ | Pandas | NumPy | Scikit-learn | Jupyter Notebook  
**Visualization:** Matplotlib | Seaborn | Folium (Interactive Maps)  
**ML Techniques:** Classification Algorithms | Feature Scaling | Statistical Analysis

---

**⭐ Star this repository if you found it helpful!**