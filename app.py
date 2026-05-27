import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")

st.title("🩺 Breast Cancer Prediction System")
st.markdown("### Early Detection using Machine Learning")

# Load models
project_root = r'C:\Users\johnpaul\Desktop\Breast-cancer-prediction'
src_path = os.path.join(project_root, 'src')

model_original = joblib.load(os.path.join(src_path, 'breast_cancer_model_original.pkl'))
model_final = joblib.load(os.path.join(src_path, 'breast_cancer_model_final.pkl'))
scaler = joblib.load(os.path.join(src_path, 'scaler.pkl'))
top_features = joblib.load(os.path.join(src_path, 'top_features.pkl'))

# Sidebar
st.sidebar.header("📋 How to Use")
st.sidebar.info("""
1. Get values from **FNA Biopsy Report**
2. Enter Mean & Worst values
3. Click **Compare Models**
4. Download report if needed
""")

st.sidebar.warning("⚠️ This is an **educational tool**. Not for real medical diagnosis.")

model_choice = st.sidebar.radio("Select Model", 
                               ["Final Model (Recommended)", "Original Model"])

if model_choice == "Final Model (Recommended)":
    selected_model = model_final
    model_name = "Final Model (Class Weights)"
else:
    selected_model = model_original
    model_name = "Original Model"

st.sidebar.markdown("---")
st.sidebar.header("🔬 Input Features")

# Input with validation
input_data = {}
for feature in top_features:
    min_val = 0.0
    max_val = 300.0 if 'area' in feature or 'perimeter' in feature else 100.0
    default = 15.0 if 'radius' in feature else 20.0 if 'perimeter' in feature else 0.1
    
    input_data[feature] = st.sidebar.number_input(
        label=feature.replace('_', ' ').title(),
        value=default,
        min_value=min_val,
        max_value=max_val,
        format="%.4f",
        help="Value from biopsy image analysis"
    )

# Buttons
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    predict_button = st.button("🚀 Compare Both Models", type="primary")
with col_btn2:
    sample_button = st.button("📊 Load Sample Data (Malignant)")

# Load Sample Data (Malignant case)
if sample_button:
    sample_values = {
        'worst concave points': 0.3,
        'worst perimeter': 150.0,
        'mean concave points': 0.25,
        'worst radius': 25.0,
        'mean perimeter': 120.0,
        'worst area': 2000.0,
        'mean radius': 20.0,
        'mean area': 1300.0,
        'mean concavity': 0.25,
        'worst concavity': 0.4,
        'mean compactness': 0.2,
        'worst compactness': 0.35
    }
    for feature in top_features:
        if feature in sample_values:
            input_data[feature] = sample_values[feature]
    st.success("Sample Malignant case loaded!")

# Prediction
if predict_button:
    try:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        pred_orig = model_original.predict(input_scaled)[0]
        prob_orig = model_original.predict_proba(input_scaled)[0][1]
        
        pred_final = model_final.predict(input_scaled)[0]
        prob_final = model_final.predict_proba(input_scaled)[0][1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Model")
            if pred_orig == 1:
                st.error(f"**MALIGNANT**")
                st.metric("Probability", f"{prob_orig:.2%}")
            else:
                st.success(f"**BENIGN**")
                st.metric("Probability", f"{(1-prob_orig):.2%}")
        
        with col2:
            st.subheader("Final Model (Recommended)")
            if pred_final == 1:
                st.error(f"**MALIGNANT**")
                st.metric("Probability", f"{prob_final:.2%}")
            else:
                st.success(f"**BENIGN**")
                st.metric("Probability", f"{(1-prob_final):.2%}")

        # Download Report
        report = f"""
Breast Cancer Prediction Report
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Model Used: {model_name}
Prediction: {'Malignant' if pred_final == 1 else 'Benign'}
Confidence: {max(prob_final, 1-prob_final):.2%}
        """
        st.download_button("📥 Download Prediction Report", report, file_name="breast_cancer_report.txt")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Feature Importance & Explanations (same as before)
st.markdown("---")
st.subheader("🔍 Feature Importance")
with st.expander("Show Feature Importance", expanded=False):
    importance = pd.DataFrame({
        'Feature': top_features,
        'Importance': abs(model_final.coef_[0])
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance.head(10), palette='viridis')
    plt.title('Top 10 Most Important Features')
    st.pyplot(fig)

st.markdown("---")
st.subheader("📘 Feature Explanations")
# Feature Explanations
with st.expander("What do these features mean?", expanded=False):
    st.markdown("""
    ### Key Biological Meanings:

    - **Radius / Perimeter / Area**: Size of the cell nucleus. **Malignant cells are usually larger**.
    - **Concavity / Concave Points**: Number and depth of dents in the nucleus border. More irregular shape = higher cancer likelihood.
    - **Mean**: Average value across all cells in the sample.
    - **Worst**: The most extreme (worst) value found in any single cell. This is often the **strongest predictor**.
    - **Texture**: Variation in gray levels (roughness).
    """)


st.caption("⚠️ **Important**: This application is for educational and research purposes only. It does not replace professional medical diagnosis.")