# AI-Breast-cancer-Predictor
A machine learning web app that predicts whether a breast tumor is malignant or benign, built with scikit-learn and Streamlit
## Live App
https://ai-breast-cancer-predictor-daep65qxhosygswcxw5pfw.streamlit.app/
## Dataset
Breast Cancer Wisconsin dataset (569 samples, 30 features) from scikit-learn.
Target: 0 = malignant, 1 = benign.
## Features used
The 5 most important features, selected using Random Forest feature importance:
- Worst area
- Worst concave points
- Mean concave points
- Worst radius
- Mean concavity
- ## Model
[Logistic Regression], with an accuracy of [0.974] on the test set.

## Files
- `app.py` - Streamlit app
- `breast_cancer_model.joblib` - trained model
- `scaler.joblib` - fitted StandardScaler
- `requirements.txt` - dependencies
- Notebook (`.ipynb`) - data exploration, training and evaluation

## Disclaimer
Educational project only, not a medical diagnosis tool.
