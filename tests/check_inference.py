import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load('diabetes_model.joblib')

# Create a dummy patient (Match the EXACT columns your model was trained on)
# I'm guessing your features based on typical diabetes datasets:
# Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
data = {
    'Pregnancies': [2],
    'Glucose': [150],
    'BloodPressure': [70],
    'SkinThickness': [20],
    'Insulin': [80],
    'BMI': [30.5],
    'DiabetesPedigreeFunction': [0.5],
    'Age': [45]
}

df = pd.DataFrame(data)

try:
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    print(f"Prediction: {prediction[0]}")
    print(f"Probability: {probability[0]}")
    print("INFERENCE SUCCESSFUL.")
except Exception as e:
    print(f"INFERENCE FAILED: {e}")
    print("Check if your feature names (columns) match the training set exactly.")