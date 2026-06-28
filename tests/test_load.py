import os
import joblib

# Define the model filename
model_path = "diabetes_model.joblib"

print("Checking if model file exists...")
if os.path.exists(model_path):
    print(f"Success: Found '{model_path}' ({os.path.getsize(model_path)} bytes).")
else:
    print(f"Error: '{model_path}' not found in the current directory.")
    print(f"Current working directory is: {os.getcwd()}")
    exit(1)

try:
    print("\nAttempting to load the model...")
    # Load the joblib file
    model = joblib.load(model_path)
    
    print("---" * 10)
    print("🎉 SUCCESS: Model loaded perfectly!")
    print("---" * 10)
    
    # Print basic model info to verify what type of object it is
    print(f"Model Object Type: {type(model)}")
    
    # If it's a scikit-learn model, show its parameters
    if hasattr(model, "get_params"):
        print("\nModel Parameters detected:")
        for param, value in list(model.get_params().items())[:5]:  # Show first 5 params
            print(f"  - {param}: {value}")
            
except Exception as e:
    print("\n❌ ERROR: Failed to load the model.")
    print(f"Details: {e}")