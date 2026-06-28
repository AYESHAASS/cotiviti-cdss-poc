import os
from groq import Groq
from dotenv import load_dotenv

# 1. Load your API Key from the .env file
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 2. Function to read your markdown file
def load_guidelines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# 3. Setup the Agent Logic
def test_agent():
    # Load the guidelines you just saved
    guidelines = load_guidelines("data/guideline.md")
    
    # Create a "Fake" patient based on your model's prediction
    # Let's pretend the model said "High Risk" for this patient
    patient_stats = "Glucose: 130 mg/dl, HbA1c: 6.0%, BMI: 29"
    prediction = "High Risk (Prediabetes)"

    prompt = f"""
    SYSTEM: You are a clinical assistant. Use ONLY the provided GUIDELINES to suggest next steps.
    
    GUIDELINES:
    {guidelines}
    
    PATIENT DATA:
    {patient_stats}
    
    ML PREDICTION:
    {prediction}
    
    TASK: Give a short recommendation. Mention the specific Table or Section from the guidelines.
    """

    # Send it to Groq
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    
    print("--- AGENT OUTPUT ---")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    test_agent()