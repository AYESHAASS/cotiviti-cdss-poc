# 🏥 Clinical Decision Support System (CDSS) with Agentic AI & Explainability

## 🎯 Overview
This Project is a **Proof of Concept (POC)** developed to address **Cotiviti Topic 2**: *Clinical Decision Making and Pattern Recognition in Health Care.*

The system bridges the gap between raw Machine Learning outputs and actionable clinical decisions. It doesn't just predict risk; it explains the "why" and provides evidence-based recommendations grounded in the **2023 Clinical Guidelines for Management of Type 2 Diabetes.**

## 🚀 Key Features
- **Classification & Prediction:** Ensemble model (XGBoost + Random Forest) providing 76%+ probability accuracy on clinical test cases.
- **Explainability (Pattern Recognition):** Integrated **SHAP (SHapley Additive exPlanations)** layer to visualize feature importance and build clinician trust.
- **Agentic Generative AI (Chain Reasoning):** A Llama-3.3-70B agent that reconciles ML predictions with stored clinical guidelines to provide safe, evidence-based recommendations.
- **TPO Alignment:** Specifically designed to optimize **Treatment, Payment, and Operations** by reducing diagnostic errors and automating policy compliance.

## 🛠 Tech Stack
- **Languages:** Python 3.11
- **Machine Learning:** Scikit-learn, XGBoost, SHAP
- **Agentic AI:** Groq API (Llama-3.3-70B-Versatile)
- **Frameworks:** Streamlit (UI), Pandas, Joblib
- **Deployment:** Docker, GitHub

## 📂 Project Structure
```text
cotiviti-cdss-poc/
├── app/
│   ├── main.py              # Streamlit UI & Logic Coordinator
│   └── explainer.py         # SHAP Explainability logic
├── guidelines/
│   └── guideline.md         # 2023 T2DM Clinical Standards (Brain)
├── model/
│   └── diabetes_model.joblib # Trained Ensemble Model
├── tests/
│   └── check_inference.py   # Unit tests for model validation
├── Dockerfile               # Containerization config
└── requirements.txt         # Dependency manifest
```

## 📊 Performance Validation
### Case: Borderline Risk Management
In edge cases where ML models may provide low-risk scores due to missing data, the **Agentic Reasoning Layer** performs a safety check against clinical thresholds (e.g., Identifying high glucose as "Prediabetes" even if the ML probability is low).

![Case 1 Analysis](assets/case%201.png)
![Case 2 Analysis](assets/case%202.png)


## ⚙️ How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AYESHAASS/cotiviti-cdss-poc
   cd cotiviti-cdss-poc
   ```
2. **Setup Environment:**
   Create a `.env` file in the root and add your Groq API Key:
   ```text
   GROQ_API_KEY=your_key_here
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the App:**
   ```bash
   streamlit run app/main.py
   ```

## 👨‍🔬 Research Background
This POC is an extension of my first-author research published in **Springer Nature**, focusing on Hybrid Deep Learning approaches for clinical decision support.

## 📺 Video Demonstration
Click the link below to watch a 4-minute walkthrough of the CDSS Proof of Concept, including the technical architecture and a live demo of the "Safety Layer" reasoning.

👉 **[Watch the Video Demo Here](https://youtu.be/i71QU5sO82kE)**
