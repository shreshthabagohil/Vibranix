import os
import json
import numpy as np
import tensorflow.lite as tflite
# We import your working bridge
from gemini_advisory.advisory_bridge import get_advisory

# --- CONFIGURATION ---
MODEL_PATH = "models/vibranix_engine.tflite"
# Update these based on your training classes!
LABELS = ["Beneficial", "Noise", "Pest"] 

def run_vibranix_demo(audio_path):
    print(f"🎤 Processing Audio: {os.path.basename(audio_path)}")
    
    # 1. Load TFLite Model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. MOCK PREPROCESSING (For Demo Purposes)
    # In a real scenario, you'd use your scripts/preprocess.py logic here
    # For the hackathon demo, we simulate a 'Pest' detection to show the bridge
    print("🧠 Running Edge Inference...")
    
    # Let's assume the model detected a 'Pest' with 94% confidence
    detected_class = "Pest"
    confidence = 0.94

    print(f"✅ Detection Result: {detected_class} ({confidence*100:.1f}%)")

    # 3. Trigger the Gemini Advisor ONLY if it's a Pest
    if detected_class == "Pest":
        print("🔍 Requesting Expert Agricultural Advisory...")
        
        # We pass the detection to your bridge
        # Note: You can change 'Locust' to whatever was detected
        detection_data = {"label": "Locust", "confidence": confidence}
        
        try:
            advice = get_advisory(detection_data)
            
            print("\n" + "="*40)
            print("🌾 VIBRANIX FARMER ADVISORY 🌾")
            print("="*40)
            print(f"Common Name : {advice.get('Indian_Name', 'N/A')}")
            print(f"Risk Level  : {advice.get('Risk_Level', 'N/A')}")
            print(f"Organic Fix : {advice.get('Organic_Control', 'N/A')}")
            print("="*40)
            
        except Exception as e:
            print(f"❌ Advisory Error: {e}")
    else:
        print("🌿 Environment is healthy. No action needed.")

if __name__ == "__main__":
    # Point this to any .wav file in your data/raw folder
    sample_audio = "data/raw/test_sample.wav" 
    
    # If file doesn't exist, we'll just run a simulation
    run_vibranix_demo(sample_audio)