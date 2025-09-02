# Mobile Phone Price Classification

A comprehensive machine learning project that predicts mobile phone price ranges based on device specifications using multiple ML algorithms and deep learning techniques.

## Project Overview

This project aims to classify mobile phones into different price categories (Low, Medium, High, Very High cost) based on their technical specifications. The model helps manufacturers, retailers, and consumers make informed decisions about mobile phone pricing strategies.

## Dataset

- **Source**: [Mobile Price Classification Dataset](https://www.kaggle.com/datasets/navjotkaushal/mobile-price-classification-dataset)
- **Target**: 4 price categories (Low, Medium, High, Very High cost)
- **Type**: Multi-class classification problem

### Features

The dataset includes the following mobile phone specifications:

| Feature | Description |
|---------|-------------|
| `Battery_power_mAh` | Total battery capacity in mAh |
| `Bluetooth` | Bluetooth support (Yes/No) |
| `Speed_of_microprocessor` | Processor clock speed |
| `Dual_sim` | Dual SIM support (Yes/No) |
| `Front_camera` | Front camera resolution in megapixels |
| `4G` | 4G network support (Yes/No) |
| `Internal_memory_gb` | Internal storage in GB |
| `Mobile_depth` | Device thickness in cm |
| `Mobile_weight` | Device weight in grams |
| `Cores_of_processor` | Number of processor cores |
| `Primary_camera` | Primary camera resolution in megapixels |
| `px_height` | Pixel resolution height |
| `Pixel_width` | Pixel resolution width |
| `Ram_mb` | RAM capacity in MB |
| `Screen_height` | Screen height in cm |
| `Screen_weight` | Screen width in cm |
| `talk_time` | Battery talk time |
| `3G` | 3G network support (Yes/No) |
| `touch_screen` | Touchscreen support (Yes/No) |
| `wifi` | WiFi support (Yes/No) |

## 🛠️ Technology Stack

### Core Libraries
- **Python 3.10+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and preprocessing

### Visualization
- **matplotlib** - Basic plotting
- **seaborn** - Statistical data visualization

### Machine Learning Models
- **CatBoost** - Gradient boosting classifier
- **Logistic Regression** - Linear classification
- **PyTorch** - Deep learning framework

### Optimization & Utilities
- **Optuna** - Hyperparameter optimization
- **kagglehub** - Dataset downloading
- **joblib** - Model serialization

## 🚀 Project Structure

```
PhonePrice-Classification/
├── PhonePrice.ipynb          # Main Jupyter notebook
├── environment.yml           # Conda environment file
├── lg_model_pipeline.joblib  # Saved best model
└── README.md                 # Project documentation
```

## Methodology

### 1. Data Preprocessing
- **Data Cleaning**: Removal of text units from numeric fields
- **Type Conversion**: Converting mixed data types to appropriate formats
- **Stratified Splitting**: Train-validation-test split maintaining class proportions

### 2. Exploratory Data Analysis (EDA)
- **Missing Values Analysis**: Verification of data completeness
- **Target Distribution**: Analysis of price category balance
- **Feature Distributions**: Histograms and box plots for all features
- **Correlation Analysis**: Heatmap of feature correlations
- **Dimensionality Reduction**: PCA visualization (2D and 3D)

### 3. Machine Learning Pipeline

#### Data Preprocessing Pipeline
- **Numerical Features**: StandardScaler for normalization
- **Categorical Features**: TargetEncoder for categorical encoding
- **Column Transformer**: Unified preprocessing pipeline

#### Model Implementation

1. **CatBoost Classifier**
   - GPU acceleration support
   - Built-in categorical feature handling
   - Early stopping mechanism
   - **Performance**: 86% accuracy

2. **Logistic Regression**
   - Scikit-learn pipeline integration
   - Hyperparameter optimization with Optuna
   - **Performance**: 95% accuracy (best model)

3. **Deep Neural Network (PyTorch)**
   - 3-layer architecture with batch normalization
   - Dropout regularization
   - Adam optimizer
   - **Performance**: 90% accuracy

### 4. Hyperparameter Optimization
- **Framework**: Optuna
- **Objective**: F1-macro score maximization
- **Search Space**: Regularization strength (C) and penalty type
- **Trials**: 50 optimization trials

### 5. Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Validation**: Stratified cross-validation
- **Feature Importance**: Analysis of top contributing features

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| **Logistic Regression** | **95%** | **95%** | **95%** | **95%** |
| Deep Neural Network | 90% | 90% | 90% | 90% |
| CatBoost | 86% | 86% | 86% | 86% |

### Best Model Performance (Optimized Logistic Regression)

| Price Category | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| High cost | 0.95 | 0.94 | 0.95 | 152 |
| Low cost | 0.97 | 0.94 | 0.96 | 155 |
| Medium cost | 0.91 | 0.94 | 0.93 | 144 |
| Very High cost | 0.97 | 0.98 | 0.98 | 149 |

**Overall Accuracy**: 95%

## Installation

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/w1thoutiq/PhonePrice-Classification.git
   cd PhonePrice-Classification
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate your_env_name
   ```

### Running the Project

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook PhonePrice.ipynb
   ```

2. **Execute cells sequentially** to:
   - Download and load the dataset
   - Perform exploratory data analysis
   - Train multiple ML models
   - Optimize hyperparameters
   - Evaluate model performance

3. **Use the trained model**
   ```python
   import joblib
   model = joblib.load('lg_model_pipeline.joblib')
   predictions = model.predict(new_data)
   ```
