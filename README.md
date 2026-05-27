# 🩺 Breast Cancer Prediction System

A robust **Machine Learning** web application that predicts whether a breast mass is **Benign** or **Malignant** based on Fine Needle Aspirate (FNA) features.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-FF6F00?style=for-the-badge&logo=scikit-learn&logoColor=white)

## ✨ Features

- **Two Models** for comparison:
  - Original Logistic Regression
  - Improved Model with Class Weights (Higher Recall)
- **Side-by-Side Prediction** comparison
- **Interactive Streamlit Web App**
- **Feature Importance Visualization**
- **Sample Data Loader** (Malignant case)
- **Prediction Report Download**
- Input validation and error handling
- Comprehensive feature explanations

## 📊 Model Performance

| Model                    | Accuracy | Recall   | Precision | F1-Score |
|-------------------------|----------|----------|-----------|----------|
| **Final Model (Recommended)** | **98.25%** | **97.62%** | 97.62%    | 97.62%   |
| Original Model           | 97.37%   | 92.86%   | 100%      | 96.30%   |

> **Recall** was prioritized because missing a cancer case (False Negative) is critical in medical diagnosis.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/Johnpaul10j/breast-cancer-prediction.git
cd breast-cancer-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app:
   Bash
   streamlit run app.py

# Project Structure

breast-cancer-prediction/
├── src/                          # Trained models & scaler
│   ├── breast_cancer_model_final.pkl
│   ├── breast_cancer_model_original.pkl
│   ├── scaler.pkl
│   └── top_features.pkl
├── notebooks/                    # EDA and Model Training
├── app.py                        # Main Streamlit Application
├── requirements.txt
└── README.md
```
# Technologies Used

Core: Python, Pandas, NumPy
ML: scikit-learn (Logistic Regression)
Visualization: Matplotlib, Seaborn
Deployment: Streamlit
Model Interpretation: Feature Importance Analysis

# Key Learnings

Handling mild class imbalance in medical datasets
Feature selection and importance analysis
Importance of Recall in healthcare ML projects
Building production-grade interactive web applications
Explainable AI (XAI) concepts

# Disclaimer
This project is for educational and research purposes only. It should not be used for real medical diagnosis. Always consult qualified medical professionals.

## Made with ❤️ by Umeh Johnpaul
