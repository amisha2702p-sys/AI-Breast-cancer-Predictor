import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('breast_cancer_model.joblib')
scaler = joblib.load('scaler.joblib')

selected_features = ['worst area', 'worst concave points', 'mean concave points', 'worst radius', 'mean concavity']
st.title("AI Breast cancer Predictor")
st.write('Enter the values below to get a prediction.')
st.markdown(
    """
    @import url('https://fonts.googleleapis.com/css2?family=Nunito:wght@600;700&display=swap');
    <style>
        .health-title{
            font-family:'Nunito', sans-serif;
            font-weight: 700;
            font-size: 42px;
            color: #0F6E8C; /*calm medical teal-blue*/
            margin-bottom: 0;
            }
        .health-subtitle {
            font-family: 'Nunito', sans-serif;
            font-weight: 600;
            font-size: 18px;
            color: #5F7A86;
            }
    </style>
    <h1 class="health-title".Patient Care Dashboard</h1>
    <p class="health-subtitle">Monitoring and insights for better health outcomes</p>
    """,
    unsafe_allow_html=True,
    )

worst_area = st.number_input('Worst Area', min_value=0.0, value=880.0, format="%.4f")
worst_cp = st.number_input('Worst Concave Points',min_value=0.0, value=0.11, format="%.4f")
mean_cp = st.number_input('Mean Concave Points', min_value=0.0, value=0.05, format="%.4f")
worst_radius = st.number_input('Worst Radius', min_value=0.0, value=16.3, format="%.4f")
mean_concavity = st.number_input('Mean Concavity', min_value=0.0, value=0.09, format="%.4f")

if st.button('Predict'):
    input_data = pd.DataFrame([[worst_area, worst_cp, mean_cp, worst_radius, mean_concavity]],
                            columns= selected_features)
    scaled_input = scaler.transform(input_data)
    result = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0]

    if result == 0:
        st.error(f'Prediction: Malignant (confidence {probability[0]:.1%})')
    else:
        st.success(f'Prediction:Benign (confidence {probability[0]:.1%})')
    
