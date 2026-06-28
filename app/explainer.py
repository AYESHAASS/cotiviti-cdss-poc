import joblib
import pandas as pd
import shap
import numpy as np

# Load the model
model = joblib.load('diabetes_model.joblib')

def get_shap_explanation(patient_df):
    # 1. Extract steps from pipeline
    classifier = model.named_steps['model']
    scaler = model.named_steps['scaler']
    
    # 2. Transform the data
    patient_scaled = scaler.transform(patient_df)
    
    # 3. Use the first estimator (Random Forest) from the VotingClassifier
    rf_model = classifier.estimators_[0]
    
    # 4. Initialize Explainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(patient_scaled)

    # 5. HANDLE SHAP OUTPUT LOGIC (The Fix)
    # If binary classification, SHAP returns a list: [class_0_probs, class_1_probs]
    # We want class 1 (High Risk)
    if isinstance(shap_values, list):
        # Take class 1, and the first (and only) patient's values
        importance_array = shap_values[1][0]
    else:
        # If it's a single array (newer SHAP versions), handle indexing
        if len(shap_values.shape) == 3: # (patients, features, classes)
            importance_array = shap_values[0, :, 1]
        else:
            importance_array = shap_values[0]

    # 6. Map to feature names
    feature_names = patient_df.columns
    feature_importance = {}
    
    for i, name in enumerate(feature_names):
        feature_importance[name] = float(importance_array[i])
    
    # 7. Sort by absolute value
    sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return sorted_importance[:3]

if __name__ == "__main__":
    # Test with a dummy patient that should have high risk
    test_patient = pd.DataFrame({
        'Pregnancies': [6],
        'Glucose': [168],
        'BloodPressure': [72],
        'SkinThickness': [35],
        'Insulin': [0],
        'BMI': [43.6],
        'DiabetesPedigreeFunction': [0.627],
        'Age': [50]
    })

    try:
        top_features = get_shap_explanation(test_patient)
        print("\n--- TOP 3 RISK FACTORS (SHAP) ---")
        for feature, val in top_features:
            influence = "INCREASED RISK" if val > 0 else "DECREASED RISK"
            print(f"{feature: <25}: {influence} ({val:.4f})")
    except Exception as e:
        print(f"STILL FAILING: {e}")