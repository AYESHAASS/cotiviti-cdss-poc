import streamlit as st
import pandas as pd
import joblib
from groq import Groq
import shap
import numpy as np
import os
from dotenv import load_dotenv

# 1. SETUP & CONFIG
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
st.set_page_config(page_title="Cotiviti CDSS POC", layout="wide")

# 2. LOAD ASSETS (Model & Guidelines)
@st.cache_resource
def load_resources():
    # Load your model and guidelines
    model = joblib.load('model/diabetes_model.joblib')
    with open("guidelines/guideline.md", "r", encoding="utf-8") as f:
     guidelines = f.read()
    return model, guidelines

model, guidelines = load_resources()

# 3. UI HEADER
st.title("🏥 Clinical Decision Support System")
st.subheader("Agentic AI for Pattern Recognition & Decision Making")
st.markdown("""
**Targeting Topic 2:** Classification, Prediction, & Agentic Generative AI for 
Treatment, Payment, & Operations (TPO).
""")
st.markdown("---")

# 4. SIDEBAR - REALISTIC INPUTS
st.sidebar.header("Patient Vital Signs")
st.sidebar.markdown("Enter patient data for risk assessment.")

with st.sidebar.form("patient_form"):
    # Section 1: Demographics
    st.subheader("Demographics")
    age = st.number_input("Age", 1, 120, 45)
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    st.divider()
    # Section 2: Clinical Vitals
    st.subheader("Clinical Vitals")
    glucose = st.number_input("Glucose (mg/dl)", 0, 300, 120)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    bp = st.number_input("Blood Pressure (mm Hg)", 0, 200, 120)
    st.divider()
    # Section 3: Advanced Biomarkers
    st.subheader("Advanced Biomarkers")
    skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47) 
    submit = st.form_submit_button("Run Clinical Analysis")

# 5. MAIN ANALYSIS PIPELINE
if submit:
    # Prepare input for the model
    input_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], 
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.header("1. Predictive Analytics")
        # ML Prediction
        prob = model.predict_proba(input_data)[0][1]
        risk_label = "HIGH RISK" if prob > 0.5 else "LOW RISK"
        color = "red" if risk_label == "HIGH RISK" else "green"
        
        st.metric("Risk Probability", f"{prob*100:.1f}%")
        st.markdown(f"Status: **:{color}[{risk_label}]**")

        # SHAP (Explainability)
        st.write("---")
        st.subheader("2. Pattern Recognition (SHAP)")
        st.write("Which factors drove this specific prediction?")
        
        classifier = model.named_steps['model']
        scaler = model.named_steps['scaler']
        patient_scaled = scaler.transform(input_data)
        
        # Using the first estimator (Random Forest) for SHAP
        explainer = shap.TreeExplainer(classifier.estimators_[0])
        shap_values = explainer.shap_values(patient_scaled)
        
        if isinstance(shap_values, list): imps = shap_values[1][0]
        else: imps = shap_values[0, :, 1] if len(shap_values.shape)==3 else shap_values[0]
        
        shap_df = pd.DataFrame({'Feature': input_data.columns, 'Impact': imps})
        shap_df = shap_df.sort_values(by='Impact', ascending=False)
        st.bar_chart(shap_df.set_index('Feature'), horizontal=True)

    with col2:
        st.header("3. Agentic Recommendation")
        with st.spinner("Agent analyzing guidelines..."):
            # Get Top 3 Drivers for the Prompt
            top_factors = shap_df.head(3)['Feature'].tolist()
            
            agent_prompt = f"""
            Identify as a Senior Clinical Decision Support Agent.
            
            PATIENT CONTEXT:
            - Glucose: {glucose} mg/dl
            - BMI: {bmi}
            - Age: {age}
            - Prediction: {risk_label} ({prob*100:.1f}% probability)
            - Top Model Drivers: {top_factors}
            
            GUIDELINE CONTEXT:
            {guidelines}
            
             TASK:
            1. Provide a clear recommendation.
            2. USE MARKDOWN: Use bold headers, bullet points, and tables. 
            3. DO NOT write long paragraphs. 
            4. Cite specific Sections from the guidelines.
            """
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": agent_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )
            
            recommendation = response.choices[0].message.content
            st.success("Analysis Complete")
            st.info(recommendation)
        
        with st.expander("View Agent's Reasoning Path"):
            st.write("The agent performed the following steps:")
            st.write("1. Extracted top features from SHAP explainability layer.")
            st.write("2. Performed zero-shot retrieval from the 2023 Clinical Guidelines.")
            st.write("3. Applied logical reasoning to reconcile ML prediction with clinical standards.")

# 6. FOOTER - BUSINESS VALUE
st.markdown("---")
st.subheader("Business Value (Cotiviti Context)")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.write("**Operational Efficiency**")
    st.caption("Automates the first pass of clinical review using evidence-based guidelines.")
with footer_col2:
    st.write("**Accuracy & Safety**")
    st.caption("Combines pattern recognition (ML) with rule-based safety nets (LLM + Guidelines).")
with footer_col3:
    st.write("**Payment Integrity**")
    st.caption("Ensures treatments align with documented clinical policies to prevent over-billing.")